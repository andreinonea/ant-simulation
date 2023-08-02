extern crate glfw;

mod gl {
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));
}

use glfw::{Action, Context, Key};
use gl::types::*;
use glam::{Mat4, Vec2, Vec3};
use rand::seq::SliceRandom;
use std::f32::consts::PI;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::str;
use std::sync::mpsc;
use std::thread;
use rand::distributions::{Distribution, Uniform};


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
    unsafe {
        let l = gl::GetUniformLocation(program, u.as_ptr());
        if l == -1 {
            println!("notyetaWARN: no location for {}", uniform)
        }
        l
    }
}

// Problem starts here.
const POINTS_OF_INTEREST: usize = 20;
static mut distance_table: [[f32; POINTS_OF_INTEREST]; POINTS_OF_INTEREST] = [[0.0f32; POINTS_OF_INTEREST]; POINTS_OF_INTEREST];

fn calculate_distance_squared(a: Vec2, b: Vec2) -> f32 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2)
}

fn point_in_circle(point: Vec2, circle: Vec2, circle_radius: f32) -> bool {
    calculate_distance_squared(point, circle) < circle_radius.powi(2)
}

fn randomize_pois(target: u32, left: f32, right: f32, bottom: f32, top: f32, exclude_radius_from_center: f32) -> Vec<Vec2> {
    let xrange = Uniform::from(left..=right);
    let yrange = Uniform::from(bottom..=top);
    let mut rng = rand::thread_rng();

    let mut pois: Vec<Vec2> = Vec::new();
    let mut count = 0;

    while count < target {
        let x = xrange.sample(&mut rng);
        let y = yrange.sample(&mut rng);

        let p = Vec2::new(x, y);

        if point_in_circle(p, Vec2::ZERO, exclude_radius_from_center) {
            continue;
        }

        pois.push(p);
        count += 1;
    }

    pois
}

fn random_path() -> Vec<u32> {
    let mut vec: Vec<u32> = (0..10).collect();
    vec.shuffle(&mut rand::thread_rng());
    vec
}

static mut best_tour_distance: f32 = f32::MAX;
static mut solution_counter: u128 = 0;

fn evaluate_distance(indices: &Vec<u32>) -> f32 {
    let mut tour_distance = 0.0f32;
    for i in 0..indices.len() {
        let next_idx = (i + 1) % indices.len();
        unsafe { tour_distance += distance_table[indices[i] as usize][indices[next_idx] as usize]; }
    }
    tour_distance
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// BRUTE FORCE

fn evaluate_solution(tx: mpsc::Sender<Vec<u32>>, points: Vec<Vec2>, indices: Vec<u32>, num_solutions: u128) {
    if indices[0] >= indices[indices.len() - 2] {
        return;
    }

    unsafe {
        solution_counter += 1;
        println!("Searched: {} / {}", solution_counter, num_solutions);
    }

    unsafe {
        let tour_distance = evaluate_distance(&indices);
        if tour_distance < best_tour_distance {
            best_tour_distance = tour_distance;
            tx.send(indices).unwrap();
        }
    }
}

fn generate_solutions(tx: mpsc::Sender<Vec<u32>>, points: Vec<Vec2>, mut indices: Vec<u32>, n: usize, num_solutions: u128) {
    if n == 1 {
        evaluate_solution(tx, points, indices, num_solutions);
    } else {
        for i in 0..n {
            generate_solutions(tx.clone(), points.clone(), indices.clone(), n - 1, num_solutions);
            let swap_index = if n % 2 == 0 { i } else { 0 };
            (indices[swap_index], indices[n - 1]) = (indices[n - 1], indices[swap_index])
        }
    }
}

fn solve_brute(tx: mpsc::Sender<Vec<u32>>, points: Vec<Vec2>, indices: Vec<u32>) {
    // loop {
    //     tx.send(random_path()).unwrap();
    //     thread::sleep(Duration::from_secs_f32(0.5f32));
    // }

    fn factorial(num: u128) -> u128 {
        (1..=num).product()
    }

    let num_solutions = factorial(indices.len() as u128 - 1) / 2;
    generate_solutions(tx, points, indices.clone(), indices.len() - 1, num_solutions);
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Ant colony optimization
// settings:
static mut should_ants_search: bool = true;
const ANTS_IN_GROUP: usize = 5;
const DIST_POWER: i32 = 4;

fn ant_journey(tx: mpsc::Sender<Vec<u32>>, start_index: u32, to_visit: &mut Vec<u32>) {
    let mut trace: Vec<u32> = Vec::new();
    let mut cur_index = start_index;

    loop {
        trace.push(cur_index);
        to_visit.retain(|&i| i != cur_index);

        let next_index = to_visit.choose_weighted(
            &mut rand::thread_rng(),
            |potential_next_index| {
                unsafe {
                    let dist = distance_table[cur_index as usize][potential_next_index.to_owned() as usize];
                    let desirability = (1.0f32 / dist).powi(DIST_POWER);
                    desirability
                }
            }
        );

        match next_index {
            Ok(next_index) => { cur_index = next_index.to_owned(); }
            _ => { break; }
        }
    }

    tx.send(trace).unwrap();
}

fn solve_ants(tx: mpsc::Sender<Vec<u32>>, points: Vec<Vec2>, indices: Vec<u32>) {
    // let start_index = indices.choose(&mut rand::thread_rng()).un;
    // let mut ant: Vec<u32> = vec![start_index];

    // let mut journey = indices.clone();
    // journey.shuffle(&mut rand::thread_rng());

    let mut total_ants = 0;
    loop  {
        let ants: Vec<u32> = indices.choose_multiple(&mut rand::thread_rng(), ANTS_IN_GROUP).cloned().collect();
        let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

        let (tx_ant, rx_ant) = mpsc::channel();

        for i in 0..ANTS_IN_GROUP {
            let start_index = ants[i];

            let mut to_visit = indices.to_owned();
            let tx_ant_i = tx_ant.clone();
            let handle = thread::spawn(move ||
                ant_journey(tx_ant_i, start_index, &mut to_visit)
            );
            handles.push(handle);
        }

        for _ in 0..ANTS_IN_GROUP {
            let handle = handles.remove(0);
            handle.join().unwrap();
        }

        for trace in rx_ant.try_iter() {
            unsafe {
                let tour_distance = evaluate_distance(&trace);
                if tour_distance < best_tour_distance {
                    best_tour_distance = tour_distance;
                    tx.send(trace).unwrap();
                }
            }
        }

        total_ants += ANTS_IN_GROUP;
        println!("Ants: {total_ants}");
    }
}

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let width = 1920;
    let height = 1080;
    let (mut window, events) = glfw.create_window(
        width,
        height,
        "Ants solve the travelling salesman problem",
        glfw::WindowMode::Windowed
    ).expect("Failed to create GLFW window.");

    window.set_key_polling(true);
    window.set_mouse_button_polling(true);
    window.set_framebuffer_size_polling(true);
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

    let circle_vertices = gen_circle(1.0f32, 28);
    let mut vao_circle: GLuint = 0;
    let mut vbo_circle: GLuint = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao_circle);
        gl::GenBuffers(1, &mut vbo_circle);

        gl::BindVertexArray(vao_circle);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_circle);

        gl::BufferData(gl::ARRAY_BUFFER,
            (circle_vertices.len() * mem::size_of::<Vec2>()) as GLsizeiptr,
            mem::transmute(&circle_vertices[0]),
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
        gl::DeleteBuffers(1, &vbo_circle);
    }

    let mut points = randomize_pois(POINTS_OF_INTEREST as u32, -150.0f32, 150.0f32, -80.0f32, 80.0f32, 40.0f32);
    let lecount = points.len();
    let mut path_indices: Vec<u32> = (0..points.len() as u32).collect();

    for a in 0..lecount {
        for b in 0..lecount {
            unsafe { distance_table[a][b] = calculate_distance_squared(points[a], points[b]); }
        }
    }

    let mut vao_path: GLuint = 0;
    let mut vbo_path: GLuint = 0;
    let mut ibo_path: GLuint = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao_path);
        gl::GenBuffers(1, &mut vbo_path);
        gl::GenBuffers(1, &mut ibo_path);

        gl::BindVertexArray(vao_path);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo_path);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo_path);

        gl::BufferData(gl::ARRAY_BUFFER,
            (points.len() * mem::size_of::<Vec2>()) as GLsizeiptr,
            mem::transmute(&points[0]),
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

        gl::BufferData(gl::ELEMENT_ARRAY_BUFFER,
            (path_indices.len() * mem::size_of::<u32>()) as GLsizeiptr,
            ptr::null(),
            gl::DYNAMIC_DRAW,
        );

        gl::BindVertexArray(0);
        gl::DeleteBuffers(1, &vbo_path);
        gl::DeleteBuffers(1, &ibo_path);

        gl::LineWidth(3.0f32);
    }

    let view = Mat4::look_at_rh(Vec3::Z * 100.0f32, Vec3::ZERO, Vec3::Y);
    let mut projection = Mat4::perspective_rh(
        90.0f32.to_radians(),
        width as f32 / height as f32,
        0.1f32,
        100.0f32
    );

    let vs = compile_shader(VS_SRC, gl::VERTEX_SHADER);
    let fs = compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
    let prog = link_program(vs, fs);

    unsafe {
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
    }

    let u_mvp_prog = get_uniform(prog, "u_mvp");
    let u_color_prog = get_uniform(prog, "u_color");

    let (tx, rx) = mpsc::channel();

    let indices = path_indices.to_owned();
    let pois = points.to_owned();
    thread::spawn(move || solve_ants(tx, pois, indices));

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event, &mut points, &view, &mut projection);
        }

        let vp = projection * view;

        unsafe {
            gl::ClearColor(0.0f32, 0.0f32, 0.0f32, 1.0f32);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::UseProgram(prog);
            gl::Uniform4f(u_color_prog, 0.2549f32, 0.4117f32, 0.8823f32, 1.0f32);

            gl::BindVertexArray(vao_circle);

            for pos in &points {
                let model = Mat4::from_translation(Vec3::new(pos.x, pos.y, 0.0f32)) * Mat4::IDENTITY;
                let mvp = vp * model;
                gl::UniformMatrix4fv(u_mvp_prog, 1, gl::FALSE, &mvp.to_cols_array()[0]);
                gl::DrawArrays(gl::TRIANGLE_FAN, 0, circle_vertices.len() as i32);
            }

            gl::BindVertexArray(vao_path);

            let indices_buffer = gl::MapBufferRange(gl::ELEMENT_ARRAY_BUFFER,
                0,
                (path_indices.len() * mem::size_of::<u32>()) as isize,
                gl::MAP_WRITE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT,
            );
            ptr::copy(
                mem::transmute(&path_indices[0]),
                indices_buffer,
                path_indices.len() * mem::size_of::<u32>(),
            );
            gl::UnmapBuffer(gl::ELEMENT_ARRAY_BUFFER);

            let model = Mat4::IDENTITY;
            let mvp = vp * model;
            gl::UniformMatrix4fv(u_mvp_prog, 1, gl::FALSE, &mvp.to_cols_array()[0]);
            gl::DrawElements(gl::LINE_LOOP, path_indices.len() as GLsizei, gl::UNSIGNED_INT, ptr::null());

            gl::BindVertexArray(0);
            gl::UseProgram(0);
        }

        match rx.try_recv() {
            Ok(res) => { path_indices = res; }
            _ => {}
        }

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteVertexArrays(1, &vao_circle);
        gl::DeleteProgram(prog);
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent, _points: &mut Vec<Vec2>, _view: &Mat4, projection: &mut Mat4) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true);
            unsafe { should_ants_search = false; }
        }
        glfw::WindowEvent::FramebufferSize(width, height) => {
            unsafe {
                gl::Viewport(0, 0, width, height);
            }
            *projection = Mat4::perspective_rh(
                90.0f32.to_radians(),
                width as f32 / height as f32,
                0.1f32,
                100.0f32
            );
        }
        _ => {}
    }
}
