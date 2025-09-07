// main.cpp — Island (polygon coast from file) + Near-Shore Foam (poly SDF) + F3 Grid (No Plateaus)
// 读取 coast.txt（或命令行参数的路径）中的2D坐标(x,y)作为海岸多边形（逆时针CCW）。
// 每行可写多对坐标：  x y   或   x,y   （允许空格/逗号/Tab混排；支持 # 与 // 注释）
// 运行：WASD/空格/左Ctrl移动，鼠标视角，TAB 捕获/释放，F1 线框，F2 背面剔除，F3 网格，F5 重新读取文本并重建。
// Deps (vcpkg): glfw3 glad glm
// Build (MSVC x64):
//   vcpkg install glfw3 glad glm
//   cl /EHsc /std:c++20 /W4 main.cpp ^
//      /I"%VCPKG_ROOT%\\installed\\x64-windows\\include" ^
//      /D_CRT_SECURE_NO_WARNINGS /MD ^
//      /Fe:island_poly_foam_grid.exe ^
//      /link /LIBPATH:"%VCPKG_ROOT%\\installed\\x64-windows\\lib" glfw3.lib glad.lib opengl32.lib

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

static void Check(bool ok, const char* msg) { if (!ok) { std::fprintf(stderr, "Error: %s\n", msg); std::exit(EXIT_FAILURE); } }

#if _DEBUG
static void APIENTRY DebugCB(GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* msg, const void*) {
    if (severity != GL_DEBUG_SEVERITY_NOTIFICATION) std::fprintf(stderr, "[GL %s] %s\n", type == GL_DEBUG_TYPE_ERROR ? "ERROR" : "MSG", msg);
}
static void EnableGLDebugIfAvailable() {
    int flags = 0; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if ((flags & GL_CONTEXT_FLAG_DEBUG_BIT) && glDebugMessageCallback) {
        glEnable(GL_DEBUG_OUTPUT); glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(DebugCB, nullptr);
        if (glDebugMessageControl) glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    }
}
#else
static void EnableGLDebugIfAvailable() {}
#endif

// --- noise ---
static uint32_t h32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
static float r01(uint32_t x) { return (x >> 8) * (1.0f / 16777216.0f); }
static float noise1(float t) {
    int i = (int)floor(t); float f = t - i;
    float a = r01(h32((uint32_t)i * 2654435761U)), b = r01(h32((uint32_t)(i + 1) * 2654435761U));
    float s = f * f * (3.f - 2.f * f); return a + (b - a) * s;
}
static float fbm1(float t, int oct = 4, float lac = 2.f, float gain = 0.5f) {
    float amp = 1, f = 1, sum = 0, n = 0; for (int i = 0; i < oct; i++) { sum += noise1(t * f) * amp; n += amp; f *= lac; amp *= gain; } return sum / (n > 0 ? n : 1);
}

// --- mesh ---
struct Vtx { glm::vec3 pos, nrm; float kind, band; };
struct Mesh {
    std::vector<Vtx> v; std::vector<uint32_t> i; GLuint vao = 0, vbo = 0, ebo = 0;
    void upload() {
        if (!vao) glGenVertexArrays(1, &vao); if (!vbo) glGenBuffers(1, &vbo); if (!ebo) glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size() * sizeof(Vtx)), v.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(i.size() * sizeof(uint32_t)), i.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, pos));
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, nrm));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, kind));
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, band));
        glBindVertexArray(0);
    }
    void destroy() { if (ebo) glDeleteBuffers(1, &ebo); if (vbo) glDeleteBuffers(1, &vbo); if (vao) glDeleteVertexArrays(1, &vao); vao = vbo = ebo = 0; }
};
static inline void addTriI(Mesh& m, uint32_t a, uint32_t b, uint32_t c) { m.i.push_back(a); m.i.push_back(b); m.i.push_back(c); }

// --- poly utils ---
static float cross2(const glm::vec2& a, const glm::vec2& b) { return a.x * b.y - a.y * b.x; }
static float area2(const std::vector<glm::vec2>& P) {
    long double A = 0; size_t n = P.size();
    for (size_t i = 0; i < n; i++) { const auto& p = P[i]; const auto& q = P[(i + 1) % n]; A += (long double)p.x * q.y - (long double)p.y * q.x; } return (float)(A * 0.5);
}
static bool isCCW(const std::vector<glm::vec2>& P) { return area2(P) > 0; }

static bool pointInTri(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
    glm::vec2 v0 = c - a, v1 = b - a, v2 = p - a;
    float d00 = glm::dot(v0, v0), d01 = glm::dot(v0, v1), d11 = glm::dot(v1, v1), d20 = glm::dot(v2, v0), d21 = glm::dot(v2, v1);
    float denom = d00 * d11 - d01 * d01; if (std::fabs(denom) < 1e-12f) return false;
    float v = (d11 * d20 - d01 * d21) / denom, w = (d00 * d21 - d01 * d20) / denom, u = 1.f - v - w;
    return (u > -1e-6f && v > -1e-6f && w > -1e-6f);
}

static void earClipTriangulate(const std::vector<glm::vec2>& poly, std::vector<uint32_t>& out) {
    size_t n = poly.size(); out.clear(); if (n < 3) return;
    if (!isCCW(poly)) { std::vector<glm::vec2> tmp = poly; std::reverse(tmp.begin(), tmp.end()); earClipTriangulate(tmp, out); return; }
    std::vector<int> V(n); for (size_t i = 0; i < n; i++) V[i] = (int)i;
    int guard = 0;
    while (V.size() > 3 && guard++ < 10000) {
        bool cut = false;
        for (size_t k = 0; k < V.size(); k++) {
            int i0 = V[(k + V.size() - 1) % V.size()], i1 = V[k], i2 = V[(k + 1) % V.size()];
            const auto& a = poly[i0]; const auto& b = poly[i1]; const auto& c = poly[i2];
            if (cross2(b - a, c - b) <= 0) continue; // 凸角
            bool any = false; for (size_t j = 0; j < V.size(); j++) { int ij = V[j]; if (ij == i0 || ij == i1 || ij == i2) continue; if (pointInTri(poly[ij], a, b, c)) { any = true; break; } }
            if (any) continue;
            out.push_back((uint32_t)i0); out.push_back((uint32_t)i1); out.push_back((uint32_t)i2);
            V.erase(V.begin() + (int)k); cut = true; break;
        }
        if (!cut) break;
    }
    if (V.size() == 3) { out.push_back((uint32_t)V[0]); out.push_back((uint32_t)V[1]); out.push_back((uint32_t)V[2]); }
}

// 射线法
static bool pointInPoly_stable(const glm::vec2& p, const std::vector<glm::vec2>& P) {
    bool c = false; int n = (int)P.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        glm::vec2 a = P[i], b = P[j];
        float dy = b.y - a.y; if (std::fabs(dy) < 1e-9f) continue;
        bool cond = ((a.y > p.y) != (b.y > p.y)) && (p.x < (b.x - a.x) * (p.y - a.y) / dy + a.x);
        if (cond) c = !c;
    }
    return c;
}

// 自定义平方长度（替代 glm::length2）
static inline float len2(const glm::vec2& v) { return glm::dot(v, v); }

// 外法线（CCW）：(-e.y, e.x)
static std::vector<glm::vec2> offsetMiter(const std::vector<glm::vec2>& P, float d, float miterLimit = 4.0f) {
    size_t n = P.size(); std::vector<glm::vec2> Q(n);
    for (size_t i = 0; i < n; i++) {
        const auto& pm = P[(i + n - 1) % n]; const auto& p = P[i]; const auto& pn = P[(i + 1) % n];
        glm::vec2 e0 = glm::normalize(p - pm), e1 = glm::normalize(pn - p);
        glm::vec2 n0 = { -e0.y, e0.x }, n1 = { -e1.y, e1.x };
        glm::vec2 m = glm::normalize(n0 + n1);
        float denom = glm::dot(m, n0); float scale = (std::fabs(denom) < 1e-6f) ? 1.f : (1.f / denom);
        scale = std::clamp(scale, -miterLimit, miterLimit);
        Q[i] = p + m * (d * scale);
    }
    return Q;
}

// 顶面法线：cross(p2-p0, p1-p0)
static void accumulateAndNormalizeTopNormals(Mesh& m, size_t baseV, size_t baseI) {
    for (size_t t = baseI; t < m.i.size(); t += 3) {
        uint32_t i0 = m.i[t], i1 = m.i[t + 1], i2 = m.i[t + 2];
        glm::vec3 p0 = m.v[i0].pos, p1 = m.v[i1].pos, p2 = m.v[i2].pos;
        glm::vec3 n = glm::normalize(glm::cross(p2 - p0, p1 - p0));
        m.v[i0].nrm += n; m.v[i1].nrm += n; m.v[i2].nrm += n;
    }
    for (size_t vi = baseV; vi < m.v.size(); ++vi) {
        glm::vec3 n = m.v[vi].nrm; if (glm::dot(n, n) < 1e-10f) n = { 0,1,0 }; m.v[vi].nrm = glm::normalize(n);
    }
}

// --- 读文件 & 预处理 ---
static void trimComment(std::string& s) {
    auto cut = s.find('#'); if (cut != std::string::npos) s.resize(cut);
    cut = s.find("//"); if (cut != std::string::npos) s.resize(cut);
}
static bool loadCoastFromFile(const char* path, std::vector<glm::vec2>& out) {
    out.clear(); std::ifstream fin(path);
    if (!fin) return false;
    std::string line;
    while (std::getline(fin, line)) {
        trimComment(line);
        for (char& c : line) if (c == ',' || c == '\t' || c == ';') c = ' ';
        std::stringstream ss(line);
        float x, y;
        while (ss >> x >> y) out.emplace_back(x, y);
    }
    // 去重相邻过近点（使用 len2）
    std::vector<glm::vec2> clean;
    const float eps2 = 1e-6f;
    for (size_t i = 0; i < out.size(); ++i) {
        if (clean.empty() || len2(out[i] - clean.back()) > eps2) clean.push_back(out[i]);
    }
    // 头尾重复去除
    if (clean.size() >= 2 && len2(clean.front() - clean.back()) <= eps2) clean.pop_back();
    out.swap(clean);
    return out.size() >= 3;
}
static void ensureCCW(std::vector<glm::vec2>& P) { if (!isCCW(P)) std::reverse(P.begin(), P.end()); }

// --- shaders ---
static const char* VS = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in float aKind;
layout(location=3) in float aBand;
uniform mat4 uMVP, uModel; uniform mat3 uNrmMat;
out vec3 vPosW; out vec3 vNrmW; out float vKind;
void main(){ vec4 pw=uModel*vec4(aPos,1.0); vPosW=pw.xyz; vNrmW=normalize(uNrmMat*aNrm); vKind=aKind; gl_Position=uMVP*vec4(aPos,1.0); }
)GLSL";

static const char* FS = R"GLSL(
#version 330 core
in vec3 vPosW; in vec3 vNrmW; in float vKind;
uniform vec3 uCam, uLightDir, uHorizon, uSky; uniform float uFogNear, uFogFar, uTime;
const int MAX_COAST=512; uniform int uCoastCount; uniform vec2 uCoast[MAX_COAST];
uniform float uFoamWidth, uFoamFreq, uWaveSpeed, uFoamDuty;
uniform float uLandRimWidth; // 地面白带宽度
// Grid uniforms
uniform int   uGridOn;        // 1=显示，0=隐藏
uniform float uGridCell;      // 小格尺寸（世界坐标）
uniform int   uGridBoldN;     // 每N格加粗
uniform float uGridMix;       // 网格混合强度 [0..1]
out vec4 FragColor;

vec3 landColor(){return vec3(0.80,0.82,0.66);}
vec3 plateauColor(){return landColor()*0.90;}
vec3 waterColor(){return vec3(0.50,0.68,0.84);}
vec3 rimColor(){return vec3(0.97,0.97,0.99);}
vec3 cliffColor(){return vec3(0.97,0.97,0.99);}

uint h32(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
float r01(uint x){ return float(x>>8u)*(1.0/16777216.0); }
float noise1(float t){ float fi=floor(t); int i=int(fi); float f=t-fi; float a=r01(h32(uint(i)*2654435761u)); float b=r01(h32(uint(i+1)*2654435761u)); float s=f*f*(3.0-2.0*f); return a+(b-a)*s; }
float fbm1_4(float t){ float amp=1.0,f=1.0,sum=0.0,n=0.0; for(int i=0;i<4;i++){ sum+=noise1(t*f)*amp; n+=amp; f*=2.0; amp*=0.5; } return sum/max(n,1e-6); }

float distSeg(vec2 p, vec2 a, vec2 b){
    vec2 ab=b-a; float t=clamp(dot(p-a,ab)/max(dot(ab,ab),1e-8),0.0,1.0);
    vec2 q=a+t*ab; return length(p-q);
}
bool pointInPoly(vec2 p){
    bool c=false; int n=uCoastCount;
    for(int i=0,j=n-1;i<n;j=i++){
        vec2 a=uCoast[i], b=uCoast[j];
        float dy=b.y-a.y; if(abs(dy)<1e-9) continue;
        bool cond=((a.y>p.y)!=(b.y>p.y)) && (p.x < (b.x-a.x)*(p.y-a.y)/dy + a.x);
        if(cond) c=!c;
    }
    return c;
}
float coastSDF(vec2 p){
    float d=1e9; for(int i=0;i<uCoastCount;i++){ vec2 a=uCoast[i], b=uCoast[(i+1)%uCoastCount]; d=min(d, distSeg(p,a,b)); }
    return pointInPoly(p)? -d : d;
}

void main(){
    vec3 base=(vKind>3.5)?cliffColor():(vKind>2.5)?rimColor():(vKind>1.5)?waterColor():(vKind>0.5)?plateauColor():landColor();
    vec3 N=normalize(vNrmW), L=normalize(-uLightDir); float diff=max(dot(N,L),0.0);
    vec3 col=base*(0.85 + (vKind>1.5?0.12:0.08)*diff);

    // ---- 海面泡沫 ----
    if(vKind>1.5 && vKind<2.5){
        vec2 xz=vPosW.xz; float s=coastSDF(xz); float nearMask=step(0.0,s)*step(s,uFoamWidth);
        float jitter=2.1*fbm1_4(dot(xz,vec2(0.21,-0.17)) + 0.12*uTime);
        float saw=fract(s*uFoamFreq + uWaveSpeed*uTime + jitter);
        float foam=step(saw, clamp(uFoamDuty,0.0,1.0))*nearMask;
        float nearSoft=smoothstep(uFoamWidth,0.0,s);
        col*= (1.0 + 0.04*nearSoft*sin(s*0.8 - 0.7*uTime + 3.0*jitter));
        col=mix(col, vec3(0.97), foam);
    }

    // ---- 地面白带（稳定可见）----
    if(vKind<0.5){
        float s = coastSDF(vPosW.xz); // 岛内为负
        float w = max(uLandRimWidth, 1e-4);
        float mask = step(-w, s) * step(s, 0.0); // -w <= s <= 0
        float soft = smoothstep(-w, -0.7*w, s) * (1.0 - smoothstep(-0.3*w, 0.0, s));
        col = mix(col, vec3(0.97,0.97,0.99), max(mask, soft*0.85));
    }

    // ---- 地图网格（F3 开关；仅地面/顶/水面）----
    if(uGridOn==1 && vKind<2.5){
        float cell = max(uGridCell, 1e-4);
        vec2 q = vPosW.xz / cell;
        vec2 fw = fwidth(q);
        vec2 a = abs(fract(q - 0.5) - 0.5) / max(fw, vec2(1e-5));
        float minorLine = 1.0 - clamp(min(a.x, a.y), 0.0, 1.0);

        float boldLine = 0.0;
        if(uGridBoldN > 1){
            float cM = cell * float(uGridBoldN);
            vec2 qM = vPosW.xz / cM;
            vec2 fwM = fwidth(qM);
            vec2 b = abs(fract(qM - 0.5) - 0.5) / max(fwM, vec2(1e-5));
            boldLine = 1.0 - clamp(min(b.x, b.y), 0.0, 1.0);
        }
        float gline = max(minorLine, boldLine);
        vec3 gridCol = vec3(0.08);
        col = mix(col, gridCol, clamp(uGridMix * gline, 0.0, 1.0));
    }

    float dist=length(uCam - vPosW), f=clamp((dist-uFogNear)/(uFogFar-uFogNear),0.0,1.0);
    vec3 fogCol=mix(uHorizon,uSky,0.3); col=mix(col,fogCol,f); FragColor=vec4(col,1.0);
}
)GLSL";

// --- shader utils ---
static GLuint mkShader(GLenum t, const char* s) {
    GLuint sh = glCreateShader(t); glShaderSource(sh, 1, &s, nullptr); glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len); std::string log(len, '\0'); glGetShaderInfoLog(sh, len, nullptr, log.data()); std::fprintf(stderr, "Shader error:\n%s\n", log.c_str()); std::exit(EXIT_FAILURE); }
    return sh;
}
static GLuint mkProgram(const char* vs, const char* fs) {
    GLuint v = mkShader(GL_VERTEX_SHADER, vs), f = mkShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len); std::string log(len, '\0'); glGetProgramInfoLog(p, len, nullptr, log.data()); std::fprintf(stderr, "Link error:\n%s\n", log.c_str()); std::exit(EXIT_FAILURE); }
    glDeleteShader(v); glDeleteShader(f); return p;
}

// --- camera & input ---
struct Cam {
    glm::vec3 pos{ 0,3.0f,9.0f }; float yaw = -90.f, pitch = -15.f, fov = 60.f, speed = 6.f, sens = 0.1f; bool first = true; double lx = 0, ly = 0;
    glm::vec3 fwd()const { float cy = std::cos(glm::radians(yaw)), sy = std::sin(glm::radians(yaw)); float cp = std::cos(glm::radians(pitch)), sp = std::sin(glm::radians(pitch)); return glm::normalize(glm::vec3(cy * cp, sp, sy * cp)); }
    glm::vec3 right()const { return glm::normalize(glm::cross(fwd(), { 0,1,0 })); }
    glm::mat4 view()const { return glm::lookAt(pos, pos + fwd(), { 0,1,0 }); }
};
static Cam gCam;
static bool gKeys[512]{}, gWire = false, gCap = true, gCull = true, gGrid = false, gReload = false;

static void keyCB(GLFWwindow* w, int k, int, int a, int) {
    if (k >= 0 && k < 512) { if (a == GLFW_PRESS) gKeys[k] = true; else if (a == GLFW_RELEASE) gKeys[k] = false; }
    if (k == GLFW_KEY_ESCAPE && a == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    if (k == GLFW_KEY_F1 && a == GLFW_PRESS) gWire = !gWire;
    if (k == GLFW_KEY_F2 && a == GLFW_PRESS) gCull = !gCull;
    if (k == GLFW_KEY_F3 && a == GLFW_PRESS) gGrid = !gGrid;             // F3: 网格开关
    if (k == GLFW_KEY_F5 && a == GLFW_PRESS) gReload = true;             // F5: 重新读取 coast.txt
    if (k == GLFW_KEY_TAB && a == GLFW_PRESS) { gCap = !gCap; glfwSetInputMode(w, GLFW_CURSOR, gCap ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL); }
}
static void cursorCB(GLFWwindow* w, double x, double y) {
    if (!gCap) { gCam.first = true; return; }
    if (gCam.first) { gCam.lx = x; gCam.ly = y; gCam.first = false; }
    double dx = x - gCam.lx, dy = gCam.ly - y; gCam.lx = x; gCam.ly = y;
    gCam.yaw += (float)dx * gCam.sens; gCam.pitch += (float)dy * gCam.sens; gCam.pitch = glm::clamp(gCam.pitch, -89.f, 89.f);
}

// --- shapes (fallback generator) ---
static std::vector<glm::vec2> makeCoastPoly(int N, float R, float) {
    std::vector<glm::vec2> P; P.reserve(N);
    for (int k = 0; k < N; k++) {
        float a = (float)k / N * glm::two_pi<float>();
        float r = R * (1.0f + 0.16f * std::sin(3 * a + 0.4f) + 0.10f * std::sin(5 * a + 1.3f) + 0.06f * std::sin(7 * a + 2.1f));
        P.push_back({ r * std::cos(a), r * std::sin(a) });
    } if (!isCCW(P)) std::reverse(P.begin(), P.end()); return P;
}

// 顶面（未使用但保留以便扩展）
static void addTopFromPoly(Mesh& m, const std::vector<glm::vec2>& P, float y, float noiseAmp, float seed, float kind) {
    size_t baseV = m.v.size();
    for (auto& q : P) {
        glm::vec3 pos(q.x, y, q.y); pos.y += noiseAmp * (fbm1(0.05f * pos.x + 0.04f * pos.z + seed) - 0.5f);
        m.v.push_back(Vtx{ pos,glm::vec3(0),kind,0 });
    }
    std::vector<uint32_t> tris; earClipTriangulate(P, tris);
    size_t baseI = m.i.size();
    for (size_t t = 0; t < tris.size(); t += 3)
        addTriI(m, (uint32_t)baseV + tris[t + 2], (uint32_t)baseV + tris[t + 1], (uint32_t)baseV + tris[t + 0]); // 反序→朝上
    accumulateAndNormalizeTopNormals(m, baseV, baseI);
}


// 水面——CCW 朝上
static void addWater(Mesh& m, float seaY) {
    float W = 100.f; glm::vec3 nW(0, 1, 0);
    uint32_t i0 = (uint32_t)m.v.size(); m.v.push_back(Vtx{ {-W,seaY - 0.04f,-W},nW,2,0 });
    uint32_t i1 = (uint32_t)m.v.size(); m.v.push_back(Vtx{ { W,seaY - 0.04f,-W},nW,2,0 });
    uint32_t i2 = (uint32_t)m.v.size(); m.v.push_back(Vtx{ {-W,seaY - 0.04f, W},nW,2,0 });
    uint32_t i3 = (uint32_t)m.v.size(); m.v.push_back(Vtx{ { W,seaY - 0.04f, W},nW,2,0 });
    addTriI(m, i0, i2, i1);
    addTriI(m, i1, i2, i3);
}

// 构建场景（无高原）
static Mesh buildSceneByPolys(const std::vector<glm::vec2>& coast, float seaY, float landY, float seed) {
    Mesh m;
    // 地面
    {
        size_t bV = m.v.size();
        for (auto& q : coast) {
            glm::vec3 pos(q.x, landY, q.y); pos.y += 0.02f * (fbm1(0.05f * pos.x + 0.04f * pos.z + seed) - 0.5f);
            m.v.push_back(Vtx{ pos,glm::vec3(0),0,0 });
        }
        std::vector<uint32_t> tris; earClipTriangulate(coast, tris);
        size_t bI = m.i.size();
        for (size_t t = 0; t < tris.size(); t += 3) addTriI(m, (uint32_t)bV + tris[t + 2], (uint32_t)bV + tris[t + 1], (uint32_t)bV + tris[t + 0]);
        accumulateAndNormalizeTopNormals(m, bV, bI);
    }
    // 水面
    addWater(m, seaY);

    m.upload(); return m;
}

int main(int argc, char** argv) {
    Check(glfwInit() != 0, "glfwInit");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Island (Polygon from File) + Foam SDF — No Plateaus", nullptr, nullptr);
    Check(win != nullptr, "create window"); glfwMakeContextCurrent(win); glfwSwapInterval(1);
    Check(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0, "glad"); EnableGLDebugIfAvailable();

    glfwSetWindowUserPointer(win, &gCam);
    glfwSetKeyCallback(win, keyCB);
    glfwSetCursorPosCallback(win, cursorCB);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    const float seaY = 0.0f, landY = 0.0f;
    const float worldR = 6.5f;

    // 读取文件或使用默认生成
    std::string coastPath = (argc > 1 ? argv[1] : "coast.txt");
    std::vector<glm::vec2> coast;
    if (loadCoastFromFile(coastPath.c_str(), coast)) {
        ensureCCW(coast);
        std::fprintf(stderr, "[info] loaded %zu points from %s\n", coast.size(), coastPath.c_str());
    }
    else {
        std::fprintf(stderr, "[warn] cannot open %s, use fallback generated polygon.\n", coastPath.c_str());
        coast = makeCoastPoly(180, worldR, 1337.0f);
    }

    Mesh scene = buildSceneByPolys(coast, seaY, landY, 1337.0f);

    GLuint prog = mkProgram(VS, FS);
    GLint uMVP = glGetUniformLocation(prog, "uMVP"), uModel = glGetUniformLocation(prog, "uModel"), uNrm = glGetUniformLocation(prog, "uNrmMat");
    GLint uCam = glGetUniformLocation(prog, "uCam"), uLight = glGetUniformLocation(prog, "uLightDir");
    GLint uHor = glGetUniformLocation(prog, "uHorizon"), uSky = glGetUniformLocation(prog, "uSky");
    GLint uNear = glGetUniformLocation(prog, "uFogNear"), uFar = glGetUniformLocation(prog, "uFogFar");
    GLint uTime = glGetUniformLocation(prog, "uTime");
    GLint uCoastCount = glGetUniformLocation(prog, "uCoastCount"), uCoastLoc = glGetUniformLocation(prog, "uCoast");
    GLint uFWidth = glGetUniformLocation(prog, "uFoamWidth"), uFFreq = glGetUniformLocation(prog, "uFoamFreq");
    GLint uWSpeed = glGetUniformLocation(prog, "uWaveSpeed"), uFDuty = glGetUniformLocation(prog, "uFoamDuty");
    GLint uRimW = glGetUniformLocation(prog, "uLandRimWidth");
    // Grid uniforms
    GLint uGridOn = glGetUniformLocation(prog, "uGridOn");
    GLint uGridCell = glGetUniformLocation(prog, "uGridCell");
    GLint uGridBoldN = glGetUniformLocation(prog, "uGridBoldN");
    GLint uGridMix = glGetUniformLocation(prog, "uGridMix");

    glEnable(GL_DEPTH_TEST);
    glFrontFace(GL_CCW);
    glEnable(GL_CULL_FACE); glCullFace(GL_BACK);

    auto t0 = std::chrono::high_resolution_clock::now();

    // 帧循环
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // F5 热重载
        if (gReload) {
            gReload = false;
            std::vector<glm::vec2> loaded;
            if (loadCoastFromFile(coastPath.c_str(), loaded)) {
                ensureCCW(loaded);
                coast.swap(loaded);
                scene.destroy();
                scene = buildSceneByPolys(coast, seaY, landY, 1337.0f);
                std::fprintf(stderr, "[info] reloaded %zu points from %s\n", coast.size(), coastPath.c_str());
            }
            else {
                std::fprintf(stderr, "[warn] reload failed: cannot open %s\n", coastPath.c_str());
            }
        }

        glm::vec3 f = gCam.fwd(), r = gCam.right(), up(0, 1, 0);
        float spd = gCam.speed * (gKeys[GLFW_KEY_LEFT_SHIFT] ? 2.f : 1.f);
        if (gKeys[GLFW_KEY_W]) gCam.pos += f * spd * 0.016f;
        if (gKeys[GLFW_KEY_S]) gCam.pos -= f * spd * 0.016f;
        if (gKeys[GLFW_KEY_A]) gCam.pos -= r * spd * 0.016f;
        if (gKeys[GLFW_KEY_D]) gCam.pos += r * spd * 0.016f;
        if (gKeys[GLFW_KEY_SPACE]) gCam.pos += up * spd * 0.016f;
        if (gKeys[GLFW_KEY_LEFT_CONTROL]) gCam.pos -= up * spd * 0.016f;

        float tt = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
        glm::vec3 lightDir = glm::normalize(glm::vec3(std::cos(tt * 0.1f) * 0.4f, -1.0f, std::sin(tt * 0.1f) * 0.4f));

        int W, H; glfwGetFramebufferSize(win, &W, &H);
        glViewport(0, 0, W, H);
        glClearColor(0.80f, 0.86f, 0.92f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glPolygonMode(GL_FRONT_AND_BACK, gWire ? GL_LINE : GL_FILL);
        if (gCull) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);

        float aspect = H > 0 ? (float)W / H : 16.f / 9.f;
        glm::mat4 P = glm::perspective(glm::radians(gCam.fov), aspect, 0.10f, 200.f);
        glm::mat4 V = gCam.view(), M(1.f), MVP = P * V * M; glm::mat3 N = glm::mat3(glm::transpose(glm::inverse(M)));

        glUseProgram(prog);
        glUniformMatrix4fv(uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
        glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(M));
        glUniformMatrix3fv(uNrm, 1, GL_FALSE, glm::value_ptr(N));
        glUniform3fv(uCam, 1, glm::value_ptr(gCam.pos));
        glUniform3fv(uLight, 1, glm::value_ptr(lightDir));
        glUniform3f(uHor, 0.76f, 0.84f, 0.90f); glUniform3f(uSky, 0.80f, 0.86f, 0.92f);
        glUniform1f(uNear, 7.0f); glUniform1f(uFar, 40.0f); glUniform1f(uTime, tt);

        // 泡沫/白带参数
        glUniform1f(uFWidth, 0.38f);
        glUniform1f(uFFreq, 2.0f);
        glUniform1f(uWSpeed, +0.5f);
        glUniform1f(uFDuty, 0.22f);
        glUniform1f(uRimW, 0.22f);

        // 将海岸坐标作为 uniform 传入（限制 MAX_COAST=512）
        const int MAX_COAST = 512;
        int n = (int)coast.size();
        if (n > MAX_COAST) {
            static std::vector<glm::vec2> tmp;
            tmp.clear();
            int step = (n + MAX_COAST - 1) / MAX_COAST;
            for (int i = 0; i < n; i += step) tmp.push_back(coast[i]);
            while ((int)tmp.size() < 3 && (int)tmp.size() < n) tmp.push_back(coast[(int)tmp.size()]);
            glUniform1i(uCoastCount, (GLint)tmp.size());
            glUniform2fv(uCoastLoc, (GLsizei)tmp.size(), (const float*)tmp.data());
        }
        else {
            glUniform1i(uCoastCount, n);
            glUniform2fv(uCoastLoc, n, (const float*)coast.data());
        }

        // 网格参数（F3）
        glUniform1i(uGridOn, gGrid ? 1 : 0);
        glUniform1f(uGridCell, 0.50f);  // 小格 0.5m
        glUniform1i(uGridBoldN, 5);     // 每 5 格加粗
        glUniform1f(uGridMix, 0.55f);   // 线条权重

        glBindVertexArray(scene.vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)scene.i.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glfwSwapBuffers(win);
    }
    scene.destroy(); glDeleteProgram(prog); glfwTerminate(); return 0;
}
