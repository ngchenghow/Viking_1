// main.cpp — Island + Plateaus with White Cliffs (CCW, cliff facing fixed)
// Deps (vcpkg): glfw3 glad glm
// Build (MSVC x64):
//   vcpkg install glfw3 glad glm
//   cl /EHsc /std:c++20 /W4 main.cpp ^
//      /I"%VCPKG_ROOT%\\installed\\x64-windows\\include" ^
//      /D_CRT_SECURE_NO_WARNINGS /MD ^
//      /Fe:island_plateau_realcliff.exe ^
//      /link /LIBPATH:"%VCPKG_ROOT%\\installed\\x64-windows\\lib" glfw3.lib glad.lib opengl32.lib

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
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
struct Vtx { glm::vec3 pos; glm::vec3 nrm; float kind; float band; }; // kind: 0=land, 2=water, 3=rim, 4=cliff
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
static inline uint32_t addV(Mesh& m, const Vtx& v) { m.v.push_back(v); return (uint32_t)m.v.size() - 1; }
static inline void addTriI(Mesh& m, uint32_t a, uint32_t b, uint32_t c) { m.i.push_back(a); m.i.push_back(b); m.i.push_back(c); }

static void addTri(Mesh& m, const Vtx& a, const Vtx& b, const Vtx& c) {
    uint32_t s = (uint32_t)m.v.size(); m.v.push_back(a); m.v.push_back(b); m.v.push_back(c);
    m.i.push_back(s); m.i.push_back(s + 1); m.i.push_back(s + 2);
}

// ---------- Shape helpers ----------
static float superellipseRadius(float angle, float rx, float rz, float n) {
    float ca = std::fabs(std::cos(angle)), sa = std::fabs(std::sin(angle));
    float k = std::pow(std::pow(ca, n) / std::pow(rx, n) + std::pow(sa, n) / std::pow(rz, n), 1.0f / n);
    return 1.0f / std::max(k, 1e-6f);
}
static float baseRadius(float a, float rx, float rz, float n,
    const std::vector<glm::vec2>& bays, float bayDepth,
    float rough, float seed)
{
    float R = superellipseRadius(a, rx, rz, n);
    for (auto b : bays) {
        float da = std::atan2(std::sin(a - b.x), std::cos(a - b.x));
        float dent = std::exp(-(da * da) / (2.0f * b.y * b.y));
        R *= (1.0f - bayDepth * dent);
    }
    float n1 = fbm1(seed + 1.5f * std::cos(a) + 0.8f * std::sin(a), 4, 2.0f, 0.55f);
    R *= (1.0f + rough * (n1 - 0.5f));
    return R;
}

struct Plateau { glm::vec2 c; float r0; float r1; float h; }; // r0 top radius; r1 unused guard

// ------- plateau: cap + vertical cliff ring -------
static void addPlateauCap(Mesh& m, const Plateau& s, float yTop, int seg = 160) {
    const float inset = 0.02f * s.r0;
    std::vector<uint32_t> ring(seg);
    glm::vec3 N(0, 1, 0);
    for (int k = 0; k < seg; ++k) {
        float a = (float)k / seg * glm::two_pi<float>();
        glm::vec2 dir(std::cos(a), std::sin(a));
        glm::vec3 p(s.c.x + dir.x * (s.r0 - inset), yTop, s.c.y + dir.y * (s.r0 - inset));
        ring[k] = addV(m, Vtx{ p, N, 0.f, 0.f });
    }
    glm::vec3 C(s.c.x, yTop, s.c.y);
    uint32_t ic = addV(m, Vtx{ C, N, 0.f, 0.f });
    for (int k = 0; k < seg; ++k) {
        int k1 = (k + 1) % seg;
        addTriI(m, ic, ring[k1], ring[k]); // CCW
    }
}
static void addPlateauCliff(Mesh& m, const Plateau& s, float yTop, float yBottom, int seg = 160) {
    std::vector<uint32_t> top(seg), bot(seg);
    for (int k = 0; k < seg; ++k) {
        float a = (float)k / seg * glm::two_pi<float>();
        glm::vec2 dir(std::cos(a), std::sin(a));
        glm::vec3 nrm(dir.x, 0, dir.y); // outward
        glm::vec3 pt(s.c.x + dir.x * s.r0, yTop, s.c.y + dir.y * s.r0);
        glm::vec3 pb(s.c.x + dir.x * s.r0, yBottom, s.c.y + dir.y * s.r0);
        top[k] = addV(m, Vtx{ pt, nrm, 4.f, 0.f });
        bot[k] = addV(m, Vtx{ pb, nrm, 4.f, 0.f });
    }
    for (int k = 0; k < seg; ++k) {
        int k1 = (k + 1) % seg;
        // >>> FIX: 使从外侧看为 CCW（原来顺序反了，导致面向相机被剔除）
        addTriI(m, top[k], top[k1], bot[k]);   // CCW
        addTriI(m, top[k1], bot[k1], bot[k]);   // CCW
    }
}

// ---------- Build island + real plateaus ----------
static Mesh buildFlatIsland(float worldR, int seg, float seaY, float landY, float seed) {
    float rx = worldR * 1.35f, rz = worldR * 1.00f, n = 3.4f;
    std::vector<glm::vec2> bays = {
        { glm::radians(-70.0f), 0.30f },
        { glm::radians(20.0f), 0.25f },
        { glm::radians(150.0f), 0.28f }
    };
    float bayDepth = 0.22f, rough = 0.10f;

    std::vector<float> Rb(seg);
    for (int k = 0; k < seg; k++) {
        float a = (float)k / seg * glm::two_pi<float>();
        Rb[k] = baseRadius(a, rx, rz, n, bays, bayDepth, rough, seed);
    }

    Mesh m;

    // island base (flat-ish)
    const int Nr = 28;
    std::vector<std::vector<uint32_t>> vid(Nr + 1, std::vector<uint32_t>(seg));
    size_t landVtxBegin = m.v.size();
    size_t landIdxBegin = m.i.size();

    for (int j = 0; j <= Nr; ++j) {
        float t = (float)j / (float)Nr;
        for (int k = 0; k < seg; ++k) {
            float a = (float)k / seg * glm::two_pi<float>();
            float R = Rb[k] * t;
            glm::vec3 p{ std::cos(a) * R, landY, std::sin(a) * R };
            float dy = 0.02f * worldR * (fbm1(0.05f * p.x + 0.04f * p.z + seed) - 0.5f);
            p.y += dy;
            vid[j][k] = addV(m, Vtx{ p, glm::vec3(0,1,0), 0.f, 0.f });
        }
    }
    for (int j = 0; j < Nr; ++j) {
        for (int k = 0; k < seg; ++k) {
            int k1 = (k + 1) % seg;
            uint32_t v00 = vid[j][k], v01 = vid[j][k1];
            uint32_t v10 = vid[j + 1][k], v11 = vid[j + 1][k1];
            addTriI(m, v00, v11, v10); // CCW
            addTriI(m, v00, v01, v11); // CCW
        }
    }
    size_t landIdxEnd = m.i.size();
    for (size_t idx = landIdxBegin; idx + 2 < landIdxEnd; idx += 3) {
        uint32_t i0 = m.i[idx], i1 = m.i[idx + 1], i2 = m.i[idx + 2];
        glm::vec3 p0 = m.v[i0].pos, p1 = m.v[i1].pos, p2 = m.v[i2].pos;
        glm::vec3 nrm = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        m.v[i0].nrm += nrm; m.v[i1].nrm += nrm; m.v[i2].nrm += nrm;
    }
    for (size_t vi = landVtxBegin; vi < m.v.size(); ++vi) {
        if (m.v[vi].kind == 0.f) {
            glm::vec3 n = m.v[vi].nrm;
            if (glm::dot(n, n) < 1e-8f) n = glm::vec3(0, 1, 0);
            m.v[vi].nrm = glm::normalize(n);
        }
    }

    // plateaus
    Plateau pts[] = {
        { { -0.28f * rx,  0.08f * rz }, 0.55f * worldR, 0.90f * worldR, 0.80f },
        { {  0.46f * rx, -0.22f * rz }, 0.42f * worldR, 0.80f * worldR, 0.55f },
        { {  0.06f * rx,  0.05f * rz }, 0.32f * worldR, 0.70f * worldR, 0.45f },
    };
    for (auto& s : pts) {
        float yTop = landY + s.h;
        float yBot = landY;
        addPlateauCap(m, s, yTop, 200);
        addPlateauCliff(m, s, yTop, yBot, 200); // CCW fixed
    }

    // shoreline rim
    float rimLift = 0.06f, rimWidth = 0.45f;
    for (int k = 0; k < seg; k++) {
        int k1 = (k + 1) % seg;
        float a0 = (float)k / seg * glm::two_pi<float>();
        float a1 = (float)k1 / seg * glm::two_pi<float>();
        glm::vec2 p0(std::cos(a0) * Rb[k], std::sin(a0) * Rb[k]);
        glm::vec2 p1(std::cos(a1) * Rb[k1], std::sin(a1) * Rb[k1]);
        glm::vec2 t2 = glm::normalize(p1 - p0);
        glm::vec2 n2(-t2.y, t2.x);
        glm::vec3 i0(p0.x, seaY + rimLift, p0.y);
        glm::vec3 i1(p1.x, seaY + rimLift, p1.y);
        glm::vec3 o0(p0.x + n2.x * rimWidth, seaY + rimLift, p0.y + n2.y * rimWidth);
        glm::vec3 o1(p1.x + n2.x * rimWidth, seaY + rimLift, p1.y + n2.y * rimWidth);
        glm::vec3 N = glm::normalize(glm::cross(i1 - i0, o0 - i0));
        Vtx A{ i0,N,3.f,0 }, B{ i1,N,3.f,0 }, C{ o0,N,3.f,0 }, D{ o1,N,3.f,0 };
        addTri(m, A, C, B);
        addTri(m, B, C, D);
    }

    // water plane
    {
        float W = worldR * 6.f, wy = seaY - 0.04f; glm::vec3 nW(0, 1, 0);
        glm::vec3 p0{ -W,wy,-W }, p1{ W,wy,-W }, p2{ -W,wy,W }, p3{ W,wy,W };
        Vtx A{ p0,nW,2.f,0 }, B{ p1,nW,2.f,0 }, C{ p2,nW,2.f,0 }, D{ p3,nW,2.f,0 };
        addTri(m, A, C, B);
        addTri(m, B, C, D);
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
vec3 cliffColor(){ return vec3(0.97,0.97,0.99); }

void main(){
    vec3 base = (vKind>3.5)? cliffColor() :
                (vKind>2.5)? rimColor()   :
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
    glm::vec3 fwd()const { float cy = std::cos(glm::radians(yaw)), sy = std::sin(glm::radians(yaw)), cp = std::cos(glm::radians(pitch)), sp = std::sin(glm::radians(pitch)); return glm::normalize(glm::vec3(cy * cp, sp, sy * cp)); }
    glm::vec3 right()const { return glm::normalize(glm::cross(fwd(), { 0,1,0 })); }
    glm::mat4 view()const { return glm::lookAt(pos, pos + fwd(), { 0,1,0 }); }
};
static bool gKeys[512]{}, gWire = false, gCap = true, gCull = true;
static void keyCB(GLFWwindow* w, int k, int, int a, int) {
    if (k >= 0 && k < 512) { if (a == GLFW_PRESS)gKeys[k] = true; else if (a == GLFW_RELEASE)gKeys[k] = false; }
    if (k == GLFW_KEY_ESCAPE && a == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    if (k == GLFW_KEY_F1 && a == GLFW_PRESS) gWire = !gWire;
    if (k == GLFW_KEY_F2 && a == GLFW_PRESS) gCull = !gCull; // toggle culling
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
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Island + Plateaus with White Cliffs — CCW (cliff fixed)", nullptr, nullptr);
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
    glFrontFace(GL_CCW);

    auto t0 = std::chrono::high_resolution_clock::now();
    double last = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        double now = glfwGetTime(); float dt = (float)(now - last); last = now;

        if (gCull) { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
        else { glDisable(GL_CULL_FACE); }

        glm::vec3 f = cam.fwd(), r = cam.right(), up(0, 1, 0);
        float spd = cam.speed * (gKeys[GLFW_KEY_LEFT_SHIFT] ? 2.f : 1.f);
        if (gKeys[GLFW_KEY_W]) cam.pos += f * spd * dt;
        if (gKeys[GLFW_KEY_S]) cam.pos -= f * spd * dt;
        if (gKeys[GLFW_KEY_A]) cam.pos -= r * spd * dt;
        if (gKeys[GLFW_KEY_D]) cam.pos += r * spd * dt;
        if (gKeys[GLFW_KEY_SPACE]) cam.pos += up * spd * dt;
        if (gKeys[GLFW_KEY_LEFT_CONTROL]) cam.pos -= up * spd * dt;

        float tt = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
        glm::vec3 lightDir = glm::normalize(glm::vec3(std::cos(tt * 0.1f) * 0.4f, -1.0f, std::sin(tt * 0.1f) * 0.4f));

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
