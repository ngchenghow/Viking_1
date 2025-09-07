// main.cpp — Flat "Bad North"-style map (no cliffs), CCW winding fixed
// Deps (vcpkg): glfw3 glad glm
// Build (MSVC x64):
//   vcpkg install glfw3 glad glm
//   cl /EHsc /std:c++20 /W4 main.cpp ^
//      /I"%VCPKG_ROOT%\\installed\\x64-windows\\include" ^
//      /D_CRT_SECURE_NO_WARNINGS /MD ^
//      /Fe:island_flat.exe ^
//      /link /LIBPATH:"%VCPKG_ROOT%\\installed\\x64-windows\\lib" glfw3.lib glad.lib opengl32.lib

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

static void Check(bool ok, const char* msg) { if (!ok) { std::fprintf(stderr, "Error: %s\n", msg); std::exit(EXIT_FAILURE); } }

#if _DEBUG
static void APIENTRY DebugCB(GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* msg, const void*) {
    if (severity != GL_DEBUG_SEVERITY_NOTIFICATION)
        std::fprintf(stderr, "[GL %s] %s\n", type == GL_DEBUG_TYPE_ERROR ? "ERROR" : "MSG", msg);
}
static void EnableGLDebugIfAvailable() {
    int flags = 0; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if ((flags & GL_CONTEXT_FLAG_DEBUG_BIT) && glDebugMessageCallback) {
        glEnable(GL_DEBUG_OUTPUT); glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(DebugCB, nullptr);
        if (glDebugMessageControl)
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    }
}
#else
static void EnableGLDebugIfAvailable() {}
#endif

// ---------- tiny 1D fbm noise ----------
static uint32_t h32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
static float r01(uint32_t x) { return (x >> 8) * (1.0f / 16777216.0f); }
static float noise1(float t) {
    int i = (int)floor(t); float f = t - i;
    float a = r01(h32((uint32_t)i * 2654435761U)), b = r01(h32((uint32_t)(i + 1) * 2654435761U));
    float s = f * f * (3.f - 2.f * f); return a + (b - a) * s;
}
static float fbm1(float t, int oct = 4, float lac = 2.f, float gain = 0.5f) {
    float amp = 1, f = 1, sum = 0, n = 0; for (int i = 0; i < oct; i++) { sum += noise1(t * f) * amp; n += amp; f *= lac; amp *= gain; }
    return sum / (n > 0 ? n : 1);
}

// ---------- Mesh ----------
struct Vtx { glm::vec3 pos; glm::vec3 nrm; float kind; float band; }; // kind: 0=land, 2=water, 3=rim
struct Mesh {
    std::vector<Vtx> v; std::vector<uint32_t> i; GLuint vao = 0, vbo = 0, ebo = 0;
    void upload() {
        if (!vao) glGenVertexArrays(1, &vao);
        if (!vbo) glGenBuffers(1, &vbo);
        if (!ebo) glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size() * sizeof(Vtx)), v.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(i.size() * sizeof(uint32_t)), i.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, pos));
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, nrm));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, kind));
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, band));
        glBindVertexArray(0);
    }
    void destroy() { if (ebo)glDeleteBuffers(1, &ebo); if (vbo)glDeleteBuffers(1, &vbo); if (vao)glDeleteVertexArrays(1, &vao); vao = vbo = ebo = 0; }
};
static void addTri(Mesh& m, const Vtx& a, const Vtx& b, const Vtx& c) {
    uint32_t s = (uint32_t)m.v.size(); m.v.push_back(a); m.v.push_back(b); m.v.push_back(c);
    m.i.push_back(s); m.i.push_back(s + 1); m.i.push_back(s + 2);
}

// ---------- Shape helpers ----------
static float superellipseRadius(float angle, float rx, float rz, float n) {
    float ca = fabs(cos(angle)), sa = fabs(sin(angle));
    float k = pow(pow(ca, n) / pow(rx, n) + pow(sa, n) / pow(rz, n), 1.0f / n);
    return 1.0f / std::max(k, 1e-6f);
}
static float baseRadius(float a, float rx, float rz, float n,
    const std::vector<glm::vec2>& bays, float bayDepth,
    float rough, float seed)
{
    float R = superellipseRadius(a, rx, rz, n);
    for (auto b : bays) {
        float da = atan2(sin(a - b.x), cos(a - b.x));
        float dent = exp(-(da * da) / (2.0f * b.y * b.y));
        R *= (1.0f - bayDepth * dent);
    }
    float n1 = fbm1(seed + 1.5f * cos(a) + 0.8f * sin(a), 4, 2.0f, 0.55f);
    R *= (1.0f + rough * (n1 - 0.5f));
    return R;
}

// ---------- Build flat island (no cliffs) ----------
static Mesh buildFlatIsland(float worldR, int seg, float seaY, float landY, float seed) {
    float rx = worldR * 1.35f, rz = worldR * 1.00f, n = 3.4f;
    std::vector<glm::vec2> bays = {
        { glm::radians(-70.0f), 0.30f },
        { glm::radians(20.0f), 0.25f },
        { glm::radians(150.0f), 0.28f }
    };
    float bayDepth = 0.22f, rough = 0.14f;

    // sample ring
    std::vector<glm::vec3> ring(seg);
    for (int k = 0; k < seg; k++) {
        float a = (float)k / seg * glm::two_pi<float>();
        float R = baseRadius(a, rx, rz, n, bays, bayDepth, rough, seed);
        ring[k] = { cos(a) * R, landY, sin(a) * R };
    }

    Mesh m;
    // top (fan) — CCW winding as seen from above
    glm::vec3 nrm(0, 1, 0), c(0, landY, 0);
    for (int k = 0; k < seg; k++) {
        int k1 = (k + 1) % seg;
        Vtx A{ c,nrm,0.f,0.f };
        Vtx B{ ring[k], nrm, 0.f, 0.f };
        Vtx C{ ring[k1],nrm, 0.f, 0.f };
        addTri(m, A, C, B);   // <— flipped to CCW
    }

    // shoreline rim (thin white ribbon) — CCW
    float rimLift = 0.06f, rimWidth = 0.45f;
    for (int k = 0; k < seg; k++) {
        int k1 = (k + 1) % seg;
        glm::vec2 p0(ring[k].x, ring[k].z), p1(ring[k1].x, ring[k1].z);
        glm::vec2 t = glm::normalize(p1 - p0);
        glm::vec2 n2(-t.y, t.x);
        glm::vec3 i0(p0.x, seaY + rimLift, p0.y);
        glm::vec3 i1(p1.x, seaY + rimLift, p1.y);
        glm::vec3 o0(p0.x + n2.x * rimWidth, seaY + rimLift, p0.y + n2.y * rimWidth);
        glm::vec3 o1(p1.x + n2.x * rimWidth, seaY + rimLift, p1.y + n2.y * rimWidth);
        glm::vec3 N = glm::normalize(glm::cross(i1 - i0, o0 - i0));
        Vtx A{ i0,N,3.f,0 }, B{ i1,N,3.f,0 }, C{ o0,N,3.f,0 }, D{ o1,N,3.f,0 };
        addTri(m, A, C, B);   // flip
        addTri(m, B, C, D);   // flip
    }

    // water plane (slightly below sea) — CCW
    {
        float W = worldR * 6.f, wy = seaY - 0.04f; glm::vec3 n(0, 1, 0);
        glm::vec3 p0{ -W,wy,-W }, p1{ W,wy,-W }, p2{ -W,wy,W }, p3{ W,wy,W };
        Vtx A{ p0,n,2.f,0 }, B{ p1,n,2.f,0 }, C{ p2,n,2.f,0 }, D{ p3,n,2.f,0 };
        addTri(m, A, C, B);   // flip
        addTri(m, B, C, D);   // flip
    }

    m.upload();
    return m;
}

// ---------- Shaders ----------
static const char* VS = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in float aKind;
layout(location=3) in float aBand;
uniform mat4 uMVP, uModel; uniform mat3 uNrmMat;
out vec3 vPosW; out vec3 vNrmW; out float vKind;
void main(){
    vec4 pw = uModel * vec4(aPos,1.0);
    vPosW = pw.xyz; vNrmW = normalize(uNrmMat * aNrm);
    vKind = aKind;
    gl_Position = uMVP * vec4(aPos,1.0);
}
)GLSL";

static const char* FS = R"GLSL(
#version 330 core
in vec3 vPosW; in vec3 vNrmW; in float vKind;
uniform vec3 uCam, uLightDir, uHorizon, uSky;
uniform float uFogNear, uFogFar;
out vec4 FragColor;

vec3 landColor(){ return vec3(0.80,0.82,0.66); }
vec3 waterColor(){ return vec3(0.50,0.68,0.84); }
vec3 rimColor()  { return vec3(0.97,0.97,0.99); }

void main(){
    vec3 base = (vKind>2.5)? rimColor() :
                (vKind>1.5)? waterColor() :
                              landColor();

    vec3 N = normalize(vNrmW);
    vec3 L = normalize(-uLightDir);
    float diff = max(dot(N,L), 0.0);
    float kFlat = 0.85;
    float kVar  = (vKind>1.5 ? 0.12 : 0.08);
    vec3 col = base * (kFlat + kVar*diff);

    float d = length(uCam - vPosW);
    float f = clamp((d - uFogNear)/(uFogFar - uFogNear), 0.0, 1.0);
    vec3 fogCol = mix(uHorizon, uSky, 0.3);
    col = mix(col, fogCol, f);

    FragColor = vec4(col,1.0);
}
)GLSL";

static GLuint mkShader(GLenum t, const char* s) {
    GLuint sh = glCreateShader(t); glShaderSource(sh, 1, &s, nullptr); glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok); if (!ok) { GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len); std::string log(len, '\0'); glGetShaderInfoLog(sh, len, nullptr, log.data()); std::fprintf(stderr, "Shader error:\n%s\n", log.c_str()); std::exit(EXIT_FAILURE); } return sh;
}
static GLuint mkProgram(const char* vs, const char* fs) {
    GLuint v = mkShader(GL_VERTEX_SHADER, vs), f = mkShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p); GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len); std::string log(len, '\0'); glGetProgramInfoLog(p, len, nullptr, log.data()); std::fprintf(stderr, "Link error:\n%s\n", log.c_str()); std::exit(EXIT_FAILURE); } glDeleteShader(v); glDeleteShader(f); return p;
}

// ---------- Camera ----------
struct Cam {
    glm::vec3 pos{ 0,3.0f,9.0f }; float yaw = -90.f, pitch = -15.f, fov = 60.f, speed = 6.f, sens = 0.1f;
    bool first = true; double lx = 0, ly = 0;
    glm::vec3 fwd()const { float cy = cos(glm::radians(yaw)), sy = sin(glm::radians(yaw)), cp = cos(glm::radians(pitch)), sp = sin(glm::radians(pitch)); return glm::normalize(glm::vec3(cy * cp, sp, sy * cp)); }
    glm::vec3 right()const { return glm::normalize(glm::cross(fwd(), { 0,1,0 })); }
    glm::mat4 view()const { return glm::lookAt(pos, pos + fwd(), { 0,1,0 }); }
};
static bool gKeys[512]{}, gWire = false, gCap = true;
static void keyCB(GLFWwindow* w, int k, int, int a, int) {
    if (k >= 0 && k < 512) { if (a == GLFW_PRESS)gKeys[k] = true; else if (a == GLFW_RELEASE)gKeys[k] = false; }
    if (k == GLFW_KEY_ESCAPE && a == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    if (k == GLFW_KEY_F1 && a == GLFW_PRESS) gWire = !gWire;
    if (k == GLFW_KEY_TAB && a == GLFW_PRESS) { gCap = !gCap; glfwSetInputMode(w, GLFW_CURSOR, gCap ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL); }
}
static void cursorCB(GLFWwindow* w, double x, double y) {
    Cam* c = (Cam*)glfwGetWindowUserPointer(w);
    if (!gCap) { c->first = true; return; }
    if (c->first) { c->lx = x; c->ly = y; c->first = false; }
    double dx = x - c->lx, dy = c->ly - y; c->lx = x; c->ly = y;
    c->yaw += (float)dx * c->sens; c->pitch += (float)dy * c->sens; c->pitch = glm::clamp(c->pitch, -89.f, 89.f);
}

// ---------- Main ----------
int main() {
    Check(glfwInit() != 0, "glfwInit");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Flat Island (no cliffs) — CCW", nullptr, nullptr);
    Check(win != nullptr, "create window");
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    Check(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0, "glad");
    EnableGLDebugIfAvailable();

    glfwSetKeyCallback(win, keyCB);
    Cam cam; glfwSetWindowUserPointer(win, &cam);
    glfwSetCursorPosCallback(win, cursorCB);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    const float seaY = 0.0f;
    const float landY = seaY + 0.10f;
    Mesh island = buildFlatIsland(
        /*worldR=*/6.5f,
        /*seg=*/  260,
        /*seaY=*/ seaY,
        /*landY=*/landY,
        /*seed=*/ 1337.0f);

    GLuint prog = mkProgram(VS, FS);
    GLint uMVP = glGetUniformLocation(prog, "uMVP"),
        uModel = glGetUniformLocation(prog, "uModel"),
        uNrm = glGetUniformLocation(prog, "uNrmMat"),
        uCam = glGetUniformLocation(prog, "uCam"),
        uLight = glGetUniformLocation(prog, "uLightDir"),
        uHor = glGetUniformLocation(prog, "uHorizon"),
        uSky = glGetUniformLocation(prog, "uSky"),
        uNear = glGetUniformLocation(prog, "uFogNear"),
        uFar = glGetUniformLocation(prog, "uFogFar");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW); // explicit

    auto t0 = std::chrono::high_resolution_clock::now();
    double last = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        double now = glfwGetTime(); float dt = (float)(now - last); last = now;

        glm::vec3 f = cam.fwd(), r = cam.right(), up(0, 1, 0);
        float spd = cam.speed * (gKeys[GLFW_KEY_LEFT_SHIFT] ? 2.f : 1.f);
        if (gKeys[GLFW_KEY_W]) cam.pos += f * spd * dt;
        if (gKeys[GLFW_KEY_S]) cam.pos -= f * spd * dt;
        if (gKeys[GLFW_KEY_A]) cam.pos -= r * spd * dt;
        if (gKeys[GLFW_KEY_D]) cam.pos += r * spd * dt;
        if (gKeys[GLFW_KEY_SPACE]) cam.pos += up * spd * dt;
        if (gKeys[GLFW_KEY_LEFT_CONTROL]) cam.pos -= up * spd * dt;

        float tt = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
        glm::vec3 lightDir = glm::normalize(glm::vec3(cos(tt * 0.1f) * 0.4f, -1.0f, sin(tt * 0.1f) * 0.4f));

        int W, H; glfwGetFramebufferSize(win, &W, &H);
        glViewport(0, 0, W, H);
        glClearColor(0.80f, 0.86f, 0.92f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, gWire ? GL_LINE : GL_FILL);

        float aspect = H > 0 ? (float)W / H : 16.f / 9.f;
        glm::mat4 P = glm::perspective(glm::radians(cam.fov), aspect, 0.10f, 200.f);
        glm::mat4 V = cam.view();
        glm::mat4 M(1.f);
        glm::mat4 MVP = P * V * M; glm::mat3 N = glm::mat3(glm::transpose(glm::inverse(M)));

        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
        glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(M));
        glUniformMatrix3fv(uNrm, 1, GL_FALSE, glm::value_ptr(N));
        glUniform3fv(uCam, 1, glm::value_ptr(cam.pos));
        glUniform3fv(uLight, 1, glm::value_ptr(lightDir));
        glUniform3f(uHor, 0.76f, 0.84f, 0.90f);
        glUniform3f(uSky, 0.80f, 0.86f, 0.92f);
        glUniform1f(uNear, 7.0f);
        glUniform1f(uFar, 40.0f);

        glBindVertexArray(island.vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)island.i.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glfwSwapBuffers(win);
    }

    island.destroy();
    glDeleteProgram(prog);
    glfwTerminate();
    return 0;
}
