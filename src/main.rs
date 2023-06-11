extern crate glfw;

mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}

use glam::Vec4;
use glfw::{Action, Context, Key};
use gl::types::*;
use glam::{Mat4,Vec2, Vec3};
use std::f32::consts::PI;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::str;


// Vertex data
static VERTEX_DATA: [GLfloat; 6] = [0.0, 0.5, 0.5, -0.5, -0.5, -0.5];

// Shader sources
static VS_SRC: &'static str = "
#version 330
layout (location = 0) in vec2 v_position;

uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(v_position, 0.0f, 1.0f);
}";

static FS_SRC: &'static str = "
#version 330

layout (location = 0) out vec4 f_color;

uniform vec4 u_color = vec4(1.0f);

void main() {
    f_color = u_color;
}";

fn compile_shader(src: &str, ty: GLenum) -> GLuint {
    let shader;
    unsafe {
        shader = gl::CreateShader(ty);
        // Attempt to compile the shader
        let c_str = CString::new(src.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        // Get the compile status
        let mut status = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetShaderInfoLog(
                shader,
                len,
                ptr::null_mut(),
                buf.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "{}",
                str::from_utf8(&buf)
                    .ok()
                    .expect("ShaderInfoLog not valid utf8")
            );
        }
    }
    shader
}

fn link_program(vs: GLuint, fs: GLuint) -> GLuint {
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);
        // Get the link status
        let mut status = gl::FALSE as GLint;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len: GLint = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetProgramInfoLog(
                program,
                len,
                ptr::null_mut(),
                buf.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "{}",
                str::from_utf8(&buf)
                    .ok()
                    .expect("ProgramInfoLog not valid utf8")
            );
        }
        program
    }
}

fn get_uniform(program: GLuint, uniform: &str) -> GLint {
    let u = CString::new(uniform.as_bytes()).unwrap();
    let mut l: GLint = -1;
    unsafe {
        l = gl::GetUniformLocation(program, u.as_ptr());
        if l == -1 {
            println!("notyetaWARN: no location for {}", uniform)
        }
    }
    l
}


fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let width = 1920;
    let height = 1080;
    let (mut window, events) = glfw.create_window(width, height, "Hello this is window", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.set_key_polling(true);
    window.make_current();

    gl::load_with(|s| window.get_proc_address(s) as *const _);

    fn gen_circle(radius: GLfloat, slices: GLuint) -> Vec<Vec2> {
        if slices < 3 {
            panic!("a circle needs at least 3 slices to be formed: given {slices}");
        }

        let mut vertices: Vec<Vec2> = Vec::new();

        // Place center vertex.
        vertices.push(Vec2::new(0.0f32, 0.0f32));

        let step = 2.0f32 * PI / slices as f32;

        for i in 0..=slices {
            let angle = i as f32 * step;
            vertices.push(Vec2::new(radius * f32::cos(angle), radius * f32::sin(angle)));
        }

        vertices
    }

    let mut points: Vec<Vec2> = Vec::new();

    let circle = gen_circle(1.0f32, 28);

    let vs = compile_shader(VS_SRC, gl::VERTEX_SHADER);
    let fs = compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
    let prog = link_program(vs, fs);

    unsafe {
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
    }

    let u_mvp_prog = get_uniform(prog, "u_mvp");
    let u_color_prog = get_uniform(prog, "u_color");

    let mut vao: GLuint = 0;
    let mut vbo: GLuint = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

        gl::BufferData(gl::ARRAY_BUFFER,
            (circle.len() * mem::size_of::<Vec2>()) as GLsizeiptr,
            mem::transmute(&circle[0]),
            gl::STATIC_DRAW,
        );

        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(
            0,
            2,
            gl::FLOAT,
            gl::FALSE,
            0,
            ptr::null(),
        );

        gl::BindVertexArray(0);
        gl::DeleteBuffers(1, &vbo);
    }

    unsafe {
        gl::Viewport(0, 0, width as GLsizei, height as GLsizei);
    }

    let view = Mat4::look_at_rh(Vec3::Z * 3.0f32, Vec3::ZERO, Vec3::Y);
    let mut projection = Mat4::perspective_rh(90.0f32, width as f32 / height as f32, 0.1f32, 100.0f32);

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event, &points, &view, &mut projection);
        }


        unsafe {
            gl::ClearColor(0.0f32, 0.0f32, 0.0f32, 1.0f32);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::UseProgram(prog);
            gl::BindVertexArray(vao);

            gl::Uniform4f(u_color_prog, 0.2549f32, 0.4117f32, 0.8823f32, 1.0f32);

            let model = Mat4::IDENTITY;
            let mvp = projection * view * model;
            gl::UniformMatrix4fv(u_mvp_prog, 1, gl::FALSE, &mvp.to_cols_array()[0]);
            gl::DrawArrays(gl::TRIANGLE_FAN, 0, circle.len() as i32);

            let model = Mat4::from_translation(Vec3::new(1.0f32, 0.0f32, 0.0f32));
            let mvp = projection * view * model;
            gl::UniformMatrix4fv(u_mvp_prog, 1, gl::FALSE, &mvp.to_cols_array()[0]);
            gl::DrawArrays(gl::TRIANGLE_FAN, 0, circle.len() as i32);

            gl::DrawArrays(gl::TRIANGLE_FAN, 0, circle.len() as i32);
            gl::DrawArrays(gl::TRIANGLE_FAN, 0, circle.len() as i32);

            gl::BindVertexArray(0);
            gl::UseProgram(0);
        }

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteProgram(prog);
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent, points: &Vec<Vec2>, view: &Mat4, projection: &mut Mat4) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        glfw::WindowEvent::FramebufferSize(width, height) => {
            unsafe {
                gl::Viewport(0, 0, width, height);
            }
            *projection = Mat4::perspective_rh(90.0f32, width as f32 / height as f32, 0.1f32, 100.0f32);
        }
        glfw::WindowEvent::Key(Key::T, _, Action::Release, _) => {
            let (x, y) = window.get_cursor_pos();
            println!("x: {x}, y: {y}");

            let final_matrix = projection.mul_mat4(view).inverse();

            let (w, h) = window.get_size();
            let ndc = Vec3::new(x as f32 / w as f32, 1.0f32 - (y as f32 / h as f32), 1.0f32) * 2.0f32 - 1.0f32;
            let homogeneous = final_matrix * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0f32);

            let world_pos = Vec3::new(homogeneous.x, homogeneous.y, homogeneous.z) / homogeneous.w;
            println!("x: {} y: {}", world_pos.x, world_pos.y);
        }
        glfw::WindowEvent::MouseButton(glfw::MouseButtonLeft, Action::Release, _) => {
            let (x, y) = window.get_cursor_pos();
            println!("x: {x}, y: {y}");
        }
        _ => {}
    }
}
