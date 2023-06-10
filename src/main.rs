extern crate glfw;

mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}

use glfw::{Action, Context, Key};
use gl::types::*;
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
in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}";

static FS_SRC: &'static str = "
#version 330
out vec4 out_color;

void main() {
    out_color = vec4(1.0, 1.0, 1.0, 1.0);
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


struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    fn new(x: f32, y: f32) -> Vec2 {
        Vec2 {x, y}
    }
}


fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let (mut window, events) = glfw.create_window(1920, 1080, "Hello this is window", glfw::WindowMode::Windowed)
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

    let circle = gen_circle(1.0f32, 32);

    let vs = compile_shader(VS_SRC, gl::VERTEX_SHADER);
    let fs = compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
    let program = link_program(vs, fs);

    unsafe {
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
    }

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

        gl::UseProgram(program);

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

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        unsafe {
            gl::BindVertexArray(vao);
            gl::ClearColor(0.0f32, 0.0f32, 0.0f32, 1.0f32);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::DrawArrays(gl::TRIANGLE_FAN, 0, circle.len() as i32);
            gl::BindVertexArray(0);
        }

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteVertexArrays(1, &vao);
        gl::DeleteProgram(program);
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        _ => {}
    }
}
