// main.cpp — Island + Plateaus with White Cliffs + Near-Shore Foam Waves
//             （CCW，泡沫曲度严格匹配海岸SDF 等距偏移曲线）
// Deps (vcpkg): glfw3 glad glm
// Build (MSVC x64):
//   vcpkg install glfw3 glad glm
//   cl /EHsc /std:c++20 /W4 main.cpp ^
//      /I"%VCPKG_ROOT%\\installed\\x64-windows\\include" ^
//      /D_CRT_SECURE_NO_WARNINGS /MD ^
//      /Fe:island_plateau_realcliff_foam.exe ^
//      /link /LIBPATH:"%VCPKG_ROOT%\\installed\\x64-windows\\lib" glfw3.lib glad.lib opengl32.lib
//
// 运行：WASD/空格/左Ctrl移动，鼠标视角，TAB 捕获/释放，F1 线框，F2 背面剔除
//
// 场景结构：
//   - 平坦岛面（轻微高度扰动）、海岸窄白边（rim）、水面
//   - 多个圆盘状高原（顶部平、侧壁陡崖为白色）
// 着色：
//   - 片元着色器内用“海岸隐式函数”的近似SDF（phi=r-R(theta)）求近岸距离 s
//   - 在 s∈[0, uFoamWidth] 区域沿法线铺设条纹（等距偏移），由 sawtooth+占空比生成泡沫
//   - 因为条纹相位仅依赖 SDF 距离，故条纹曲线与海岸曲率保持一致（严格平行）
//   - 轻微 FBM 抖动只影响相位不改变曲率（不会扭曲条纹的几何平行性）

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

// 简易断言：失败则打印并退出
static void Check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "Error: %s\n", msg); std::exit(EXIT_FAILURE); }
}

#if _DEBUG
// —— OpenGL Debug 输出：便于定位GL错误，仅在 Debug 下启用 ——
static void APIENTRY DebugCB(GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* msg, const void*) {
    if (severity != GL_DEBUG_SEVERITY_NOTIFICATION)
        std::fprintf(stderr, "[GL %s] %s\n", type == GL_DEBUG_TYPE_ERROR ? "ERROR" : "MSG", msg);
}
static void EnableGLDebugIfAvailable() {
    int flags = 0; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if ((flags & GL_CONTEXT_FLAG_DEBUG_BIT) && glDebugMessageCallback) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(DebugCB, nullptr);
        if (glDebugMessageControl)
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    }
}
#else
static void EnableGLDebugIfAvailable() {}
#endif

// ======================= 轻量 1D fBm 噪声（用于地表、水面细微扰动/抖动） =======================
// 整数哈希：将 32 位整数打散到均匀分布（确定性随机）
static uint32_t h32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
// 将整型映射到 [0,1) 浮点
static float r01(uint32_t x) { return (x >> 8) * (1.0f / 16777216.0f); }
// 1D value-noise：线性插值（用 smootherstep 平滑）
static float noise1(float t) {
    int i = (int)floor(t); float f = t - i;
    float a = r01(h32((uint32_t)i * 2654435761U));
    float b = r01(h32((uint32_t)(i + 1) * 2654435761U));
    float s = f * f * (3.f - 2.f * f);      // smootherstep
    return a + (b - a) * s;
}
// 1D fBm：叠加多频噪声，返回 0~1 左右的平滑值
static float fbm1(float t, int oct = 4, float lac = 2.f, float gain = 0.5f) {
    float amp = 1, f = 1, sum = 0, n = 0;
    for (int i = 0; i < oct; i++) { sum += noise1(t * f) * amp; n += amp; f *= lac; amp *= gain; }
    return sum / (n > 0 ? n : 1);
}

// ======================= 基础网格数据结构（带 vao/vbo/ebo + 上传/销毁） =======================
struct Vtx { glm::vec3 pos; glm::vec3 nrm; float kind; float band; };
// kind 用于着色器内部快速分材质：0=地面 1=高原平顶 2=水面 3=海岸窄白带rim 4=高原侧壁（白色陡崖）
struct Mesh {
    std::vector<Vtx> v; std::vector<uint32_t> i; GLuint vao = 0, vbo = 0, ebo = 0;
    // 上传到 GPU
    void upload() {
        if (!vao) glGenVertexArrays(1, &vao);
        if (!vbo) glGenBuffers(1, &vbo);
        if (!ebo) glGenBuffers(1, &ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size() * sizeof(Vtx)), v.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(i.size() * sizeof(uint32_t)), i.data(), GL_STATIC_DRAW);
        // layout(location=0..3) 对应 VS 中的 aPos/aNrm/aKind/aBand
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, pos));
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, nrm));
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, kind));
        glEnableVertexAttribArray(3); glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, band));
        glBindVertexArray(0);
    }
    // 释放 GPU 资源
    void destroy() {
        if (ebo) glDeleteBuffers(1, &ebo);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (vao) glDeleteVertexArrays(1, &vao);
        vao = vbo = ebo = 0;
    }
};
static inline uint32_t addV(Mesh& m, const Vtx& v) { m.v.push_back(v); return (uint32_t)m.v.size() - 1; }
static inline void addTriI(Mesh& m, uint32_t a, uint32_t b, uint32_t c) { m.i.push_back(a); m.i.push_back(b); m.i.push_back(c); }
static void addTri(Mesh& m, const Vtx& a, const Vtx& b, const Vtx& c) {
    uint32_t s = (uint32_t)m.v.size(); m.v.push_back(a); m.v.push_back(b); m.v.push_back(c);
    m.i.push_back(s); m.i.push_back(s + 1); m.i.push_back(s + 2);
}

// ======================= 海岸形状参数化（超椭圆 + 海湾凹陷 + 周期粗糙） =======================
// 超椭圆半径：|x/rx|^n + |z/rz|^n = 1 的极坐标半径 R(theta)
static float superellipseRadius(float angle, float rx, float rz, float n) {
    float ca = std::fabs(std::cos(angle)), sa = std::fabs(std::sin(angle));
    float k = std::pow(std::pow(ca, n) / std::pow(rx, n) + std::pow(sa, n) / std::pow(rz, n), 1.0f / n);
    return 1.0f / std::max(k, 1e-6f);
}
// 基础半径 Rb(theta)：超椭圆 * 若干高斯“海湾”凹陷 * 周期性粗糙（fbm）
static float baseRadius(
    float a, float rx, float rz, float n,
    const std::vector<glm::vec2>& bays, float bayDepth,
    float rough, float seed
) {
    float R = superellipseRadius(a, rx, rz, n);
    // 海湾凹陷：以高斯 dent 在角度域制造凹陷（b.x=角度中心, b.y=角宽）
    for (auto b : bays) {
        float da = std::atan2(std::sin(a - b.x), std::cos(a - b.x));   // wrap 到 [-pi,pi]
        float dent = std::exp(-(da * da) / (2.0f * b.y * b.y));
        R *= (1.0f - bayDepth * dent);
    }
    // 周期粗糙：对 R 再乘上 1 + rough*(fbm-0.5)
    float n1 = fbm1(seed + 1.5f * std::cos(a) + 0.8f * std::sin(a), 4, 2.0f, 0.55f);
    R *= (1.0f + rough * (n1 - 0.5f));
    return R;
}

// ======================= 圆盘高原（顶/侧壁）生成 =======================
struct Plateau { glm::vec2 c; float r0; float r1; float h; };
// 高原顶部盖片（kind=1，法线=Y+）
static void addPlateauCap(Mesh& m, const Plateau& s, float yTop, int seg = 160) {
    const float inset = 0.02f * s.r0;                // 稍微内缩避免与侧壁共面重合
    std::vector<uint32_t> ring(seg);
    glm::vec3 N(0, 1, 0);
    for (int k = 0; k < seg; ++k) {
        float a = (float)k / seg * glm::two_pi<float>();
        glm::vec2 dir(std::cos(a), std::sin(a));
        glm::vec3 p(s.c.x + dir.x * (s.r0 - inset), yTop, s.c.y + dir.y * (s.r0 - inset));
        ring[k] = addV(m, Vtx{ p,N,1.f,0.f });
    }
    glm::vec3 C(s.c.x, yTop, s.c.y);
    uint32_t ic = addV(m, Vtx{ C,N,1.f,0.f });
    // 以中心扇形三角形缝合
    for (int k = 0; k < seg; ++k) { int k1 = (k + 1) % seg; addTriI(m, ic, ring[k1], ring[k]); }
}
// 高原侧壁（kind=4，法线朝外，白色陡崖）
static void addPlateauCliff(Mesh& m, const Plateau& s, float yTop, float yBottom, int seg = 160) {
    std::vector<uint32_t> top(seg), bot(seg);
    for (int k = 0; k < seg; ++k) {
        float a = (float)k / seg * glm::two_pi<float>();
        glm::vec2 dir(std::cos(a), std::sin(a));
        glm::vec3 nrm(dir.x, 0, dir.y);                      // 水平外法线
        glm::vec3 pt(s.c.x + dir.x * s.r0, yTop, s.c.y + dir.y * s.r0);
        glm::vec3 pb(s.c.x + dir.x * s.r0, yBottom, s.c.y + dir.y * s.r0);
        top[k] = addV(m, Vtx{ pt,nrm,4.f,0.f });
        bot[k] = addV(m, Vtx{ pb,nrm,4.f,0.f });
    }
    // 拉直墙两三角
    for (int k = 0; k < seg; ++k) {
        int k1 = (k + 1) % seg;
        addTriI(m, top[k], top[k1], bot[k]);
        addTriI(m, top[k1], bot[k1], bot[k]);
    }
}

// ======================= 整个岛屿（地面 + 海岸rim + 水面 + 高原）生成 =======================
static Mesh buildFlatIsland(float worldR, int seg, float seaY, float landY, float seed) {
    // —— 海岸基础形状参数（超椭圆 + 海湾） ——
    float rx = worldR * 1.35f, rz = worldR * 1.00f, n = 3.4f;
    std::vector<glm::vec2> bays = {
        { glm::radians(-70.0f), 0.30f },
        { glm::radians(20.0f),  0.25f },
        { glm::radians(150.0f), 0.28f }
    };
    float bayDepth = 0.22f;      // 海湾凹陷强度
    float rough = 0.10f;         // 周期粗糙强度

    // 预采样 Rb(theta) 以便 CPU 侧构网
    std::vector<float> Rb(seg);
    for (int k = 0; k < seg; k++) {
        float a = (float)k / seg * glm::two_pi<float>();
        Rb[k] = baseRadius(a, rx, rz, n, bays, bayDepth, rough, seed);
    }

    Mesh m;

    // —— 地面：同心环带三角带（顶部略微高度扰动，法线后续平均） ——
    const int Nr = 28;  // 向心段数（越大越圆滑）
    std::vector<std::vector<uint32_t>> vid(Nr + 1, std::vector<uint32_t>(seg));
    size_t landVtxBegin = m.v.size();
    size_t landIdxBegin = m.i.size();

    for (int j = 0; j <= Nr; ++j) {
        float t = (float)j / (float)Nr;                 // 半径比例
        for (int k = 0; k < seg; ++k) {
            float a = (float)k / seg * glm::two_pi<float>();
            float R = Rb[k] * t;
            glm::vec3 p{ std::cos(a) * R, landY, std::sin(a) * R };
            // 轻微高度扰动（不改变法线大致方向）
            float dy = 0.02f * worldR * (fbm1(0.05f * p.x + 0.04f * p.z + seed) - 0.5f);
            p.y += dy;
            vid[j][k] = addV(m, Vtx{ p, glm::vec3(0,1,0), 0.f, 0.f }); // kind=0 地面
        }
    }
    // 按环带缝合三角
    for (int j = 0; j < Nr; ++j) {
        for (int k = 0; k < seg; ++k) {
            int k1 = (k + 1) % seg;
            uint32_t v00 = vid[j][k], v01 = vid[j][k1];
            uint32_t v10 = vid[j + 1][k], v11 = vid[j + 1][k1];
            addTriI(m, v00, v11, v10);
            addTriI(m, v00, v01, v11);
        }
    }
    // 对地面法线做平均（邻三角累加，最后归一化）
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

    // —— 多个圆盘高原（顶部+侧壁） ——
    Plateau pts[] = {
        {{-0.28f * rx,  0.08f * rz}, 0.55f * worldR, 0.90f * worldR, 0.80f}, // c, r0, r1(未用), h
        {{ 0.46f * rx, -0.22f * rz}, 0.42f * worldR, 0.80f * worldR, 0.55f},
        {{ 0.06f * rx,  0.05f * rz}, 0.32f * worldR, 0.70f * worldR, 0.45f},
    };
    for (auto& s : pts) {
        float yTop = landY + s.h, yBot = landY;
        addPlateauCap(m, s, yTop, 200);
        addPlateauCliff(m, s, yTop, yBot, 200);
    }

    // —— 海岸窄白带 rim（kind=3）：顺着海岸等距外扩一小条（视觉上是“岸线高光边”） ——
    float rimLift = 0.06f;     // 抬高到地面之上，避免与地面 z-fight
    float rimWidth = 0.12f;    // 径向外扩宽度
    int segN = (int)Rb.size();
    for (int k = 0; k < segN; k++) {
        int k1 = (k + 1) % segN;
        float a0 = (float)k / segN * glm::two_pi<float>();
        float a1 = (float)k1 / segN * glm::two_pi<float>();
        glm::vec2 p0(std::cos(a0) * Rb[k], std::sin(a0) * Rb[k]);
        glm::vec2 p1(std::cos(a1) * Rb[k1], std::sin(a1) * Rb[k1]);
        glm::vec2 t2 = glm::normalize(p1 - p0);       // 切向量
        glm::vec2 n2(-t2.y, t2.x);                    // 外法线（2D）
        glm::vec3 i0(p0.x, landY + rimLift, p0.y);
        glm::vec3 i1(p1.x, landY + rimLift, p1.y);
        glm::vec3 o0(p0.x + n2.x * rimWidth, landY + rimLift, p0.y + n2.y * rimWidth);
        glm::vec3 o1(p1.x + n2.x * rimWidth, landY + rimLift, p1.y + n2.y * rimWidth);
        glm::vec3 N = glm::normalize(glm::cross(i1 - i0, o0 - i0)); // 近似上向法线
        Vtx A{ i0,N,3.f,0 }, B{ i1,N,3.f,0 }, C{ o0,N,3.f,0 }, D{ o1,N,3.f,0 };
        addTri(m, A, C, B); addTri(m, B, C, D);
    }

    // —— 水面（kind=2）：一块大矩形，片元里用 SDF 判断“近岸距离”并画泡沫 ——
    {
        float W = worldR * 6.f;
        float wy = seaY - 0.04f;             // 稍微压低，避免与地面穿插
        glm::vec3 nW(0, 1, 0);
        glm::vec3 p0{ -W,wy,-W }, p1{ W,wy,-W }, p2{ -W,wy,W }, p3{ W,wy,W };
        Vtx A{ p0,nW,2.f,0 }, B{ p1,nW,2.f,0 }, C{ p2,nW,2.f,0 }, D{ p3,nW,2.f,0 };
        addTri(m, A, C, B); addTri(m, B, C, D);
    }

    m.upload();
    return m;
}

// ======================= 顶点 & 片元着色器 =======================
// 顶点：仅做常规变换，输出世界坐标与世界法线（用于光照），以及 kind
static const char* VS = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in float aKind;
layout(location=3) in float aBand;

uniform mat4 uMVP, uModel;
uniform mat3 uNrmMat;

out vec3 vPosW;
out vec3 vNrmW;
out float vKind;

void main(){
    vec4 pw = uModel * vec4(aPos,1.0);
    vPosW = pw.xyz;
    vNrmW = normalize(uNrmMat * aNrm);
    vKind = aKind;
    gl_Position = uMVP * vec4(aPos,1.0);
}
)GLSL";

// === 片元着色器：海岸隐式函数近似 SDF，令泡沫条纹是海岸的等距偏移曲线（曲率一致） ===
static const char* FS = R"GLSL(
#version 330 core
in vec3 vPosW;
in vec3 vNrmW;
in float vKind;

uniform vec3 uCam;
uniform vec3 uLightDir;
uniform vec3 uHorizon;
uniform vec3 uSky;
uniform float uFogNear;
uniform float uFogFar;

// —— 近岸泡沫相关（与 CPU 侧参数一致，以保证 coastSDF 一致性）——
uniform float uTime;        // 时间
uniform float uRx, uRz, uNExp;
uniform int   uBayCount;
uniform vec2  uBays[8];     // (angle, width) 海湾参数
uniform float uBayDepth;
uniform float uRough;
uniform float uSeed;

uniform float uFoamWidth;   // 近岸泡沫作用宽度（海侧，沿 SDF 法线方向的真实距离）
uniform float uFoamFreq;    // 条/米（沿 SDF 法线方向测量）
uniform float uWaveSpeed;   // 泡沫相位沿法线移动速度：>0 往岸移动（s 减小）
uniform float uFoamDuty;    // 占空比（0~1），决定白条宽度

out vec4 FragColor;

// ———— 基础配色（简洁mono风格） ————
vec3 landColor(){    return vec3(0.80,0.82,0.66); }
vec3 plateauColor(){ return landColor() * 0.90; }
vec3 waterColor(){   return vec3(0.50,0.68,0.84); }
vec3 rimColor()  {   return vec3(0.97,0.97,0.99); }
vec3 cliffColor(){   return vec3(0.97,0.97,0.99); }

// ———— 与 CPU 一致的 1D fBm，用于轻微抖动相位（不改变曲率，仅扰动条纹位置） ————
uint h32(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
float r01(uint x){ return float(x>>8u)*(1.0/16777216.0); }
float noise1(float t){
    float fi=floor(t); int i=int(fi); float f=t-fi;
    float a=r01(h32(uint(i)*2654435761u));
    float b=r01(h32(uint(i+1)*2654435761u));
    float s=f*f*(3.0-2.0*f);
    return a+(b-a)*s;
}
float fbm1_4(float t){
    float amp=1.0,f=1.0,sum=0.0,n=0.0;
    for(int i=0;i<4;i++){ sum+=noise1(t*f)*amp; n+=amp; f*=2.0; amp*=0.5; }
    return sum/max(n,1e-6);
}

// ———— 超椭圆 + 海湾 + 周期粗糙，与 CPU 的 baseRadius 等价 ————
float superellipseRadius(float ang, float rx, float rz, float nexp){
    float ca=abs(cos(ang)), sa=abs(sin(ang));
    float rxn=pow(rx,nexp), rzn=pow(rz,nexp);
    float k=pow(pow(ca,nexp)/rxn + pow(sa,nexp)/rzn, 1.0/nexp);
    return 1.0/max(k,1e-6);
}
float baseRadiusGL(float ang){
    float R=superellipseRadius(ang,uRx,uRz,uNExp);
    for(int i=0;i<uBayCount;i++){
        vec2 b=uBays[i];
        float da=atan(sin(ang-b.x),cos(ang-b.x));          // wrap 到 [-pi,pi]
        float dent=exp(-(da*da)/(2.0*b.y*b.y));
        R*= (1.0 - uBayDepth*dent);
    }
    float n1=fbm1_4(uSeed + 1.5*cos(ang) + 0.8*sin(ang));
    R*= (1.0 + uRough*(n1-0.5));
    return R;
}

// ———— 数值导数 R'(theta)（对角度做中心差分） ————
float dRb_dAng(float ang){
    float h = 0.0045;                                   // 小步长（经验值）
    return (baseRadiusGL(ang+h) - baseRadiusGL(ang-h)) / (2.0*h);
}

// ———— 近似 SDF：phi(x,z) = r - R(theta)；sdf = phi / |∇phi| ————
//   推导：r = sqrt(x^2+z^2), theta = atan2(z,x)
//   ∂phi/∂x = ∂r/∂x - R'(theta)*∂theta/∂x
//           = x/r - R'(theta)*(-z/r^2)
//           = x/r + (z/r^2)*R'(theta)
//   ∂phi/∂z = z/r - R'(theta)*( x/r^2)
//           = z/r - (x/r^2)*R'(theta)
//   故 ∇phi = (x/r + (z/r^2)R',  z/r - (x/r^2)R')
float coastSDF(vec2 xz){
    float x = xz.x, z = xz.y;
    float r2 = max(dot(xz,xz), 1e-6);
    float r  = sqrt(r2);
    float ang = atan(z,x);
    float R  = baseRadiusGL(ang);
    float dR = dRb_dAng(ang);
    vec2 g = vec2(x/r + (z/r2)*dR,
                  z/r - (x/r2)*dR);
    float phi = r - R;                                  // phi>0：在海侧；phi=0：海岸线
    return phi / max(length(g), 1e-6);                  // 归一化为“真实距离”（近似）
}

void main(){
    // 基础颜色（按 kind）
    vec3 base=(vKind>3.5)?cliffColor():
              (vKind>2.5)?rimColor():
              (vKind>1.5)?waterColor():
              (vKind>0.5)?plateauColor():landColor();

    // 简单定向漫反射（略压平，避免高光跳变）
    vec3 N=normalize(vNrmW);
    vec3 L=normalize(-uLightDir);
    float diff=max(dot(N,L),0.0);
    float kFlat=0.85;
    float kVar =(vKind>1.5?0.12:0.08);
    vec3 col=base*(kFlat + kVar*diff);

    // —— 仅对水面（kind∈(1.5,2.5)）叠加“近岸泡沫”逻辑 ——
    if(vKind>1.5 && vKind<2.5){
        vec2 xz=vPosW.xz;

        // s = 近似“到海岸的法向距离”，s>0 表示海侧
        float s = coastSDF(xz);

        // 仅在海侧且处于泡沫作用宽度内才渲染白泡
        float nearMask = step(0.0,s) * step(s, uFoamWidth);

        // 轻微抖动（改变条纹相位但不改变几何曲率）
        float nJitter = fbm1_4(dot(xz,vec2(0.22,-0.17)) + 0.15*uTime);

        // 条纹相位：cycles = s*freq + speed*t + jitter
        // 重要：s 按 SDF “真实距离”，因此等值线严格平行海岸
        float cycles = s*uFoamFreq + uWaveSpeed*uTime + 2.2*nJitter;
        float saw = fract(cycles);                       // 0..1 锯齿相位
        float foam = step(saw, clamp(uFoamDuty,0.0,1.0)) * nearMask;

        // 近岸水色起伏（幅度在近岸最大，远处趋于 0）
        float nearSoft = smoothstep(uFoamWidth, 0.0, s); // 近岸1，远处0
        float waterUndulate = 1.0 + 0.04 * nearSoft * sin(s*0.8 - 0.7*uTime + 3.0*nJitter);
        col *= waterUndulate;

        // Mono 风格：将泡沫区域混到“白色”
        col = mix(col, rimColor(), foam);
    }

    // 简单雾
    float dist=length(uCam - vPosW);
    float f=clamp((dist - uFogNear)/(uFogFar - uFogNear),0.0,1.0);
    vec3 fogCol=mix(uHorizon,uSky,0.3);
    col=mix(col,fogCol,f);

    FragColor=vec4(col,1.0);
}
)GLSL";

// 创建编译/链接着色器小工具
static GLuint mkShader(GLenum t, const char* s) {
    GLuint sh = glCreateShader(t); glShaderSource(sh, 1, &s, nullptr); glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetShaderInfoLog(sh, len, nullptr, log.data());
        std::fprintf(stderr, "Shader error:\n%s\n", log.c_str()); std::exit(EXIT_FAILURE);
    } return sh;
}
static GLuint mkProgram(const char* vs, const char* fs) {
    GLuint v = mkShader(GL_VERTEX_SHADER, vs), f = mkShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetProgramInfoLog(p, len, nullptr, log.data());
        std::fprintf(stderr, "Link error:\n%s\n", log.c_str()); std::exit(EXIT_FAILURE);
    } glDeleteShader(v); glDeleteShader(f); return p;
}

// ======================= 简单第一人称相机 + 输入 =======================
struct Cam {
    glm::vec3 pos{ 0,3.0f,9.0f }; float yaw = -90.f, pitch = -15.f, fov = 60.f, speed = 6.f, sens = 0.1f;
    bool first = true; double lx = 0, ly = 0;
    glm::vec3 fwd()const {
        float cy = std::cos(glm::radians(yaw)), sy = std::sin(glm::radians(yaw));
        float cp = std::cos(glm::radians(pitch)), sp = std::sin(glm::radians(pitch));
        return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
    }
    glm::vec3 right()const { return glm::normalize(glm::cross(fwd(), { 0,1,0 })); }
    glm::mat4 view()const { return glm::lookAt(pos, pos + fwd(), { 0,1,0 }); }
};
static bool gKeys[512]{}, gWire = false, gCap = true, gCull = true;
// 键盘：WASD移动、空格/LeftCtrl 上下、F1 线框、F2 背剔、Tab 捕获鼠标
static void keyCB(GLFWwindow* w, int k, int, int a, int) {
    if (k >= 0 && k < 512) { if (a == GLFW_PRESS) gKeys[k] = true; else if (a == GLFW_RELEASE) gKeys[k] = false; }
    if (k == GLFW_KEY_ESCAPE && a == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    if (k == GLFW_KEY_F1 && a == GLFW_PRESS) gWire = !gWire;
    if (k == GLFW_KEY_F2 && a == GLFW_PRESS) gCull = !gCull;
    if (k == GLFW_KEY_TAB && a == GLFW_PRESS) {
        gCap = !gCap;
        glfwSetInputMode(w, GLFW_CURSOR, gCap ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
}
// 鼠标视角
static void cursorCB(GLFWwindow* w, double x, double y) {
    Cam* c = (Cam*)glfwGetWindowUserPointer(w);
    if (!gCap) { c->first = true; return; }
    if (c->first) { c->lx = x; c->ly = y; c->first = false; }
    double dx = x - c->lx, dy = c->ly - y; c->lx = x; c->ly = y;
    c->yaw += (float)dx * c->sens; c->pitch += (float)dy * c->sens;
    c->pitch = glm::clamp(c->pitch, -89.f, 89.f);
}

int main() {
    // —— 初始化窗口/上下文/GL函数 ——
    Check(glfwInit() != 0, "glfwInit");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1280, 720,
        "Island + Plateaus + Near-Shore Foam — CCW (SDF-curved foam)", nullptr, nullptr);
    Check(win != nullptr, "create window");
    glfwMakeContextCurrent(win); glfwSwapInterval(1);                // VSYNC=1
    Check(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0, "glad");
    EnableGLDebugIfAvailable();

    // ———— 输入绑定 ————
    glfwSetKeyCallback(win, keyCB);
    Cam cam; glfwSetWindowUserPointer(win, &cam);
    glfwSetCursorPosCallback(win, cursorCB);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // —— 基准高度：海平面 & 陆地 —— 
    const float seaY = 0.0f;
    const float landY = seaY + 0.10f;

    // —— 海岸形状参数（与 FS 同步） ——
    const float worldR = 6.5f;
    const float rx = worldR * 1.35f, rz = worldR * 1.00f, nExp = 3.4f;
    const glm::vec2 bays[3] = {
        { glm::radians(-70.0f), 0.30f },
        { glm::radians(20.0f),  0.25f },
        { glm::radians(150.0f), 0.28f }
    };
    const int bayCount = 3;
    const float bayDepth = 0.22f;
    const float rough = 0.10f;
    const float seed = 1337.0f;

    // —— 生成场景网格 —— 
    Mesh island = buildFlatIsland(worldR, 260, seaY, landY, seed);

    // —— 着色器 & uniform 位置 —— 
    GLuint prog = mkProgram(VS, FS);
    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uModel = glGetUniformLocation(prog, "uModel");
    GLint uNrm = glGetUniformLocation(prog, "uNrmMat");
    GLint uCam = glGetUniformLocation(prog, "uCam");
    GLint uLight = glGetUniformLocation(prog, "uLightDir");
    GLint uHor = glGetUniformLocation(prog, "uHorizon");
    GLint uSky = glGetUniformLocation(prog, "uSky");
    GLint uNear = glGetUniformLocation(prog, "uFogNear");
    GLint uFar = glGetUniformLocation(prog, "uFogFar");

    // 近岸浪花相关 uniform
    GLint uTime = glGetUniformLocation(prog, "uTime");
    GLint uRxLoc = glGetUniformLocation(prog, "uRx");
    GLint uRzLoc = glGetUniformLocation(prog, "uRz");
    GLint uNExpLoc = glGetUniformLocation(prog, "uNExp");
    GLint uBayCnt = glGetUniformLocation(prog, "uBayCount");
    GLint uBaysLoc = glGetUniformLocation(prog, "uBays");
    GLint uBayDep = glGetUniformLocation(prog, "uBayDepth");
    GLint uRoughL = glGetUniformLocation(prog, "uRough");
    GLint uSeedL = glGetUniformLocation(prog, "uSeed");
    GLint uFWidth = glGetUniformLocation(prog, "uFoamWidth");
    GLint uFFreq = glGetUniformLocation(prog, "uFoamFreq");
    GLint uWSpeed = glGetUniformLocation(prog, "uWaveSpeed");
    GLint uFDuty = glGetUniformLocation(prog, "uFoamDuty");

    // —— 全局渲染状态 —— 
    glEnable(GL_DEPTH_TEST);
    glFrontFace(GL_CCW);        // 和构网一致：逆时针为正面

    auto t0 = std::chrono::high_resolution_clock::now();
    double last = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        // ———— 输入 & 时间 ————
        glfwPollEvents();
        double now = glfwGetTime(); float dt = (float)(now - last); last = now;

        // 背面剔除开关（F2）
        if (gCull) { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
        else glDisable(GL_CULL_FACE);

        // 相机 WASD + 上下
        glm::vec3 f = cam.fwd(), r = cam.right(), up(0, 1, 0);
        float spd = cam.speed * (gKeys[GLFW_KEY_LEFT_SHIFT] ? 2.f : 1.f);
        if (gKeys[GLFW_KEY_W]) cam.pos += f * spd * dt;
        if (gKeys[GLFW_KEY_S]) cam.pos -= f * spd * dt;
        if (gKeys[GLFW_KEY_A]) cam.pos -= r * spd * dt;
        if (gKeys[GLFW_KEY_D]) cam.pos += r * spd * dt;
        if (gKeys[GLFW_KEY_SPACE]) cam.pos += up * spd * dt;
        if (gKeys[GLFW_KEY_LEFT_CONTROL]) cam.pos -= up * spd * dt;

        // 光源缓慢转动
        float tt = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
        glm::vec3 lightDir = glm::normalize(glm::vec3(std::cos(tt * 0.1f) * 0.4f, -1.0f, std::sin(tt * 0.1f) * 0.4f));

        // 视口 & 清屏 & 线框（F1）
        int W, H; glfwGetFramebufferSize(win, &W, &H);
        glViewport(0, 0, W, H);
        glClearColor(0.80f, 0.86f, 0.92f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, gWire ? GL_LINE : GL_FILL);

        // 矩阵
        float aspect = H > 0 ? (float)W / H : 16.f / 9.f;
        glm::mat4 P = glm::perspective(glm::radians(cam.fov), aspect, 0.10f, 200.f);
        glm::mat4 V = cam.view();
        glm::mat4 M(1.f);
        glm::mat4 MVP = P * V * M;
        glm::mat3 N = glm::mat3(glm::transpose(glm::inverse(M)));

        // 绑定 shader & uniform
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

        // —— 将海岸参数同步给 FS，保证 coastSDF 一致 —— 
        glUniform1f(uTime, tt);
        glUniform1f(uRxLoc, rx);
        glUniform1f(uRzLoc, rz);
        glUniform1f(uNExpLoc, nExp);
        glUniform1i(uBayCnt, bayCount);
        glUniform2fv(uBaysLoc, bayCount, (const float*)bays);
        glUniform1f(uBayDep, bayDepth);
        glUniform1f(uRoughL, rough);
        glUniform1f(uSeedL, seed);

        // —— 近岸泡沫参数（可按需调节）——
        glUniform1f(uFWidth, 0.38f);  // 泡沫条纹离岸最大距离（米）
        glUniform1f(uFFreq, 2.0f);    // 每米 2 条（法向）
        glUniform1f(uWSpeed, -0.5f);  // 速度：FS 约定 >0 往岸，这里设 -0.5f（往外），可改为 +0.5f 往岸
        glUniform1f(uFDuty, 0.22f);   // 白条占空比（宽度）
        glUniform1f(uBayDep, 0.22f);  // 再同步一次，便于快速试参

        // 绘制
        glBindVertexArray(island.vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)island.i.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glfwSwapBuffers(win);
    }

    // —— 资源回收 ——
    island.destroy();
    glDeleteProgram(prog);
    glfwTerminate();
    return 0;
}
