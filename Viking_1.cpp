// main.cpp — Island + Plateaus with White Cliffs (CCW, cliff facing fixed, darker plateau)
// 说明：生成“Bad North”风格的岛屿：平坦浅绿色陆地 + 白色海岸边缘 + 多个高原（顶部略深绿、侧壁白色陡崖） + 蓝色水面。
//      采用 CCW 正面约定，真实侧壁网格（非着色假象），摄像机可自由漫游。
// Deps (vcpkg): glfw3 glad glm
// Build (MSVC x64):
//   vcpkg install glfw3 glad glm
//   cl /EHsc /std:c++20 /W4 main.cpp ^
//      /I"%VCPKG_ROOT%\\installed\\x64-windows\\include" ^
//      /D_CRT_SECURE_NO_WARNINGS /MD ^
//      /Fe:island_plateau_realcliff.exe ^
//      /link /LIBPATH:"%VCPKG_ROOT%\\installed\\x64-windows\\lib" glfw3.lib glad.lib opengl32.lib
//
// 运行时快捷键：
//   ESC  退出
//   WASD / 空格 / 左Ctrl  移动
//   鼠标 视角（TAB 捕获/释放鼠标）
//   F1   线框/填充 切换
//   F2   背面剔除 开/关
//
// 主要模块：
//   1) 轻量 1D fBm 噪声：用于岛的微小起伏与轮廓粗糙
//   2) 网格封装：VAO/VBO/EBO 上传与绘制索引
//   3) 基于超椭圆 + 海湾凹口 + 粗糙度的岛形轮廓
//   4) 平坦陆地(带微起伏) + 海岸白色“rim” + 水面
//   5) “真实”高原：顶部圆盘 + 竖直白色悬崖环（CCW 朝外）
//   6) 简单着色器：按 kind 区分材质颜色 + 漫反射 + 线性雾

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

// 小工具：断言检查
static void Check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "Error: %s\n", msg); std::exit(EXIT_FAILURE); }
}

#if _DEBUG
// —— OpenGL 调试输出回调（仅 Debug 构建尝试启用）——
static void APIENTRY DebugCB(GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* msg, const void*) {
    // 过滤掉“通知”级别，只打印消息与错误
    if (severity != GL_DEBUG_SEVERITY_NOTIFICATION)
        std::fprintf(stderr, "[GL %s] %s\n", type == GL_DEBUG_TYPE_ERROR ? "ERROR" : "MSG", msg);
}
static void EnableGLDebugIfAvailable() {
    int flags = 0; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    // 若上下文支持调试，且 glad 找到了回调入口，则启用
    if ((flags & GL_CONTEXT_FLAG_DEBUG_BIT) && glDebugMessageCallback) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(DebugCB, nullptr);
        if (glDebugMessageControl)
            // 关闭通知级别，保留更重要的级别
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    }
}
#else
static void EnableGLDebugIfAvailable() {}
#endif

// ====================== 1) 轻量 1D fBm 噪声 ======================
// 说明：用于制造地表微弱起伏，和轮廓半径的细微粗糙变化。
//       这里采用极小代价的哈希+线性插值噪声，再叠加多倍频。

// 32-bit 散列（Avalanche 风格）
static uint32_t h32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}
// 将哈希整数映射到 [0,1) 浮点
static float r01(uint32_t x) {
    return (x >> 8) * (1.0f / 16777216.0f); // 24-bit 精度
}
// 1D 值噪声：按 t 的整数段与小数段，线性插值两个随机值（平滑步进）
static float noise1(float t) {
    int i = (int)floor(t); float f = t - i;
    float a = r01(h32((uint32_t)i * 2654435761U));
    float b = r01(h32((uint32_t)(i + 1) * 2654435761U));
    // 平滑插值（smoothstep）
    float s = f * f * (3.f - 2.f * f);
    return a + (b - a) * s;
}
// 简单 fBm：oct 倍频叠加，频率 *lac，振幅 *gain
static float fbm1(float t, int oct = 4, float lac = 2.f, float gain = 0.5f) {
    float amp = 1, f = 1, sum = 0, n = 0;
    for (int i = 0; i < oct; i++) {
        sum += noise1(t * f) * amp;
        n   += amp;
        f   *= lac;
        amp *= gain;
    }
    return sum / (n > 0 ? n : 1);
}

// ====================== 2) 网格封装 ======================
// 顶点属性说明：
//   pos  位置
//   nrm  法线（世界空间）
//   kind 类型：0=陆地、1=高原顶部、2=水面、3=海岸边（rim）、4=悬崖
//   band 预留（本例未用，可做分层/条纹等）
struct Vtx { glm::vec3 pos; glm::vec3 nrm; float kind; float band; };

struct Mesh {
    std::vector<Vtx> v;              // 顶点数组
    std::vector<uint32_t> i;         // 索引数组
    GLuint vao = 0, vbo = 0, ebo = 0;// OpenGL 资源句柄

    // 上传数据到 GPU
    void upload() {
        if (!vao) glGenVertexArrays(1, &vao);
        if (!vbo) glGenBuffers(1, &vbo);
        if (!ebo) glGenBuffers(1, &ebo);

        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(v.size() * sizeof(Vtx)), v.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(i.size() * sizeof(uint32_t)), i.data(), GL_STATIC_DRAW);

        // 位置 (location=0)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, pos));
        // 法线 (location=1)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, nrm));
        // kind (location=2)
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, kind));
        // band (location=3)
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)offsetof(Vtx, band));

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

// 便捷添加：返回新顶点索引
static inline uint32_t addV(Mesh& m, const Vtx& v) {
    m.v.push_back(v); return (uint32_t)m.v.size() - 1;
}
// 便捷添加：三角形索引
static inline void addTriI(Mesh& m, uint32_t a, uint32_t b, uint32_t c) {
    m.i.push_back(a); m.i.push_back(b); m.i.push_back(c);
}
// 便捷添加：直接塞入 3 顶点与索引（顺序为 CCW）
static void addTri(Mesh& m, const Vtx& a, const Vtx& b, const Vtx& c) {
    uint32_t s = (uint32_t)m.v.size();
    m.v.push_back(a); m.v.push_back(b); m.v.push_back(c);
    m.i.push_back(s); m.i.push_back(s + 1); m.i.push_back(s + 2);
}

// ====================== 3) 轮廓工具：超椭圆 + 海湾凹口 + 粗糙度 ======================

// 超椭圆半径：给定角度、x/z 轴半径 rx/rz、指数 n（n>2 越接近矩形，=2 是椭圆）
static float superellipseRadius(float angle, float rx, float rz, float n) {
    float ca = std::fabs(std::cos(angle));
    float sa = std::fabs(std::sin(angle));
    // |x/rx|^n + |z/rz|^n = 1  →  r = 1 / ((|cos|^n/rx^n + |sin|^n/rz^n)^(1/n))
    float k = std::pow(std::pow(ca, n) / std::pow(rx, n) + std::pow(sa, n) / std::pow(rz, n), 1.0f / n);
    return 1.0f / std::max(k, 1e-6f);
}

// 基础半径：超椭圆 + 多个海湾凹口 + 粗糙度噪声
static float baseRadius(
    float a, float rx, float rz, float n,                   // 超椭圆参数
    const std::vector<glm::vec2>& bays, float bayDepth,     // 海湾中心角/宽度集，及凹深
    float rough, float seed                                 // 轮廓粗糙度与种子
) {
    // 1) 超椭圆基形
    float R = superellipseRadius(a, rx, rz, n);

    // 2) 沿角度分布的若干“海湾”凹陷（高斯钟形降低 R）
    for (auto b : bays) {
        // b.x = 海湾中心角，b.y = 海湾角宽（标准差近似）
        float da = std::atan2(std::sin(a - b.x), std::cos(a - b.x)); // wrap 到 [-pi,pi]
        float dent = std::exp(-(da * da) / (2.0f * b.y * b.y));
        R *= (1.0f - bayDepth * dent);
    }

    // 3) 细微粗糙度：fBm(角度映射)
    float n1 = fbm1(seed + 1.5f * std::cos(a) + 0.8f * std::sin(a), 4, 2.0f, 0.55f);
    R *= (1.0f + rough * (n1 - 0.5f));

    return R;
}

// ====================== 4) 高原：顶部圆盘 + 垂直悬崖环 ======================

struct Plateau {
    glm::vec2 c; // 平面中心 (x,z)
    float r0;    // 顶部半径
    float r1;    // 预留（防护/过渡用，本例未使用）
    float h;     // 相对陆地的高度（世界单位）
};

// 顶部“盖子”网格（圆盘，kind=1）
static void addPlateauCap(Mesh& m, const Plateau& s, float yTop, int seg = 160) {
    const float inset = 0.02f * s.r0;        // 略微内缩，避免和侧壁接缝重合
    std::vector<uint32_t> ring(seg);
    glm::vec3 N(0, 1, 0);                    // 顶部法线朝上
    for (int k = 0; k < seg; ++k) {
        float a = (float)k / seg * glm::two_pi<float>();
        glm::vec2 dir(std::cos(a), std::sin(a));
        glm::vec3 p(s.c.x + dir.x * (s.r0 - inset), yTop, s.c.y + dir.y * (s.r0 - inset));
        ring[k] = addV(m, Vtx{ p, N, 1.f, 0.f }); // kind=1 → 高原顶部（片元颜色略深绿）
    }
    // 圆心
    glm::vec3 C(s.c.x, yTop, s.c.y);
    uint32_t ic = addV(m, Vtx{ C, N, 1.f, 0.f });

    // 以 CCW 生成扇形三角形
    for (int k = 0; k < seg; ++k) {
        int k1 = (k + 1) % seg;
        addTriI(m, ic, ring[k1], ring[k]); // CCW，保证正面朝上
    }
}

// 垂直悬崖环（外表面），kind=4（白色）
static void addPlateauCliff(Mesh& m, const Plateau& s, float yTop, float yBottom, int seg = 160) {
    std::vector<uint32_t> top(seg), bot(seg);
    for (int k = 0; k < seg; ++k) {
        float a = (float)k / seg * glm::two_pi<float>();
        glm::vec2 dir(std::cos(a), std::sin(a));

        // 侧壁外法线：沿半径方向
        glm::vec3 nrm(dir.x, 0, dir.y);

        // 顶圈、底圈
        glm::vec3 pt(s.c.x + dir.x * s.r0, yTop,    s.c.y + dir.y * s.r0);
        glm::vec3 pb(s.c.x + dir.x * s.r0, yBottom, s.c.y + dir.y * s.r0);

        top[k] = addV(m, Vtx{ pt, nrm, 4.f, 0.f });
        bot[k] = addV(m, Vtx{ pb, nrm, 4.f, 0.f });
    }
    // 以 CCW 拼接外表面（面向外环），两三角组成一四边形
    for (int k = 0; k < seg; ++k) {
        int k1 = (k + 1) % seg;
        addTriI(m, top[k],  top[k1], bot[k]);   // CCW
        addTriI(m, top[k1], bot[k1], bot[k]);   // CCW
    }
}

// ====================== 5) 构建岛屿 + 高原 + 海岸 + 水面 ======================

static Mesh buildFlatIsland(float worldR, int seg, float seaY, float landY, float seed) {
    // 超椭圆长短轴 + 指数（形状控制）
    float rx = worldR * 1.35f, rz = worldR * 1.00f, n = 3.4f;

    // 海湾参数：若干角度位置与角宽（弧度）
    std::vector<glm::vec2> bays = {
        { glm::radians(-70.0f), 0.30f },
        { glm::radians(20.0f),  0.25f },
        { glm::radians(150.0f), 0.28f }
    };
    float bayDepth = 0.22f; // 海湾凹陷深度（对半径的相对比例）
    float rough    = 0.10f; // 轮廓粗糙度

    // 预计算每个角段的基础半径 Rb[k]
    std::vector<float> Rb(seg);
    for (int k = 0; k < seg; k++) {
        float a = (float)k / seg * glm::two_pi<float>();
        Rb[k] = baseRadius(a, rx, rz, n, bays, bayDepth, rough, seed);
    }

    Mesh m;

    // ---------- (A) 平坦岛面（带微起伏） ----------
    // 思路：用极坐标同心环采样（Nr 个半径层 × seg 个角段）。
    const int Nr = 28; // 半径层数
    std::vector<std::vector<uint32_t>> vid(Nr + 1, std::vector<uint32_t>(seg));

    // 记录陆地部分的顶点/索引范围，便于之后进行“逐三角累加法线”
    size_t landVtxBegin = m.v.size();
    size_t landIdxBegin = m.i.size();

    // 顶点：每层半径按 t=0..1 与 Rb[k] 线性缩放，y 为 landY + 轻微噪声起伏
    for (int j = 0; j <= Nr; ++j) {
        float t = (float)j / (float)Nr; // 半径层的插值参数
        for (int k = 0; k < seg; ++k) {
            float a = (float)k / seg * glm::two_pi<float>();
            float R = Rb[k] * t;
            glm::vec3 p{ std::cos(a) * R, landY, std::sin(a) * R };

            // 给地面少量起伏（幅度约 worldR*0.02）
            float dy = 0.02f * worldR * (fbm1(0.05f * p.x + 0.04f * p.z + seed) - 0.5f);
            p.y += dy;

            vid[j][k] = addV(m, Vtx{ p, glm::vec3(0,1,0), 0.f, 0.f }); // 初始法线随意（稍后会重新计算）
        }
    }

    // 索引：相邻两圈拼三角形（注意 CCW）
    for (int j = 0; j < Nr; ++j) {
        for (int k = 0; k < seg; ++k) {
            int k1 = (k + 1) % seg;
            uint32_t v00 = vid[j][k],   v01 = vid[j][k1];
            uint32_t v10 = vid[j+1][k], v11 = vid[j+1][k1];
            addTriI(m, v00, v11, v10); // CCW
            addTriI(m, v00, v01, v11); // CCW
        }
    }

    // 计算“陆地”部分的平均顶点法线：逐三角累加法线，再归一化
    size_t landIdxEnd = m.i.size();
    for (size_t idx = landIdxBegin; idx + 2 < landIdxEnd; idx += 3) {
        uint32_t i0 = m.i[idx], i1 = m.i[idx + 1], i2 = m.i[idx + 2];
        glm::vec3 p0 = m.v[i0].pos, p1 = m.v[i1].pos, p2 = m.v[i2].pos;
        glm::vec3 nrm = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        m.v[i0].nrm += nrm; m.v[i1].nrm += nrm; m.v[i2].nrm += nrm;
    }
    // 归一化陆地顶点法线（避免零向量）
    for (size_t vi = landVtxBegin; vi < m.v.size(); ++vi) {
        if (m.v[vi].kind == 0.f) {
            glm::vec3 n = m.v[vi].nrm;
            if (glm::dot(n, n) < 1e-8f) n = glm::vec3(0, 1, 0);
            m.v[vi].nrm = glm::normalize(n);
        }
    }

    // ---------- (B) 多个高原：顶部 + 悬崖环 ----------
    Plateau pts[] = {
        //    中心(x,z)                     顶部半径    备用r1      高度h
        { { -0.28f * rx,  0.08f * rz }, 0.55f * worldR, 0.90f * worldR, 0.80f },
        { {  0.46f * rx, -0.22f * rz }, 0.42f * worldR, 0.80f * worldR, 0.55f },
        { {  0.06f * rx,  0.05f * rz }, 0.32f * worldR, 0.70f * worldR, 0.45f },
    };
    for (auto& s : pts) {
        float yTop = landY + s.h; // 顶部高度 = 陆地高度 + h
        float yBot = landY;       // 悬崖底部接到陆地
        addPlateauCap  (m, s, yTop, 200); // 顶部圆盘，kind=1（略深绿，易与陆地区分）
        addPlateauCliff(m, s, yTop, yBot, 200); // 垂直白色悬崖环，CCW 外表面
    }

    // ---------- (C) 海岸“rim”白边 ----------
    // 思路：沿岛轮廓外扩一点点，做一圈微抬高的白色带子（kind=3），模拟浪线/白沙
    float rimLift  = 0.06f; // 抬升高度
    float rimWidth = 0.45f; // 外扩宽度
    for (int k = 0; k < seg; k++) {
        int k1 = (k + 1) % seg;

        float a0 = (float)k  / seg * glm::two_pi<float>();
        float a1 = (float)k1 / seg * glm::two_pi<float>();

        glm::vec2 p0(std::cos(a0) * Rb[k],  std::sin(a0) * Rb[k]);
        glm::vec2 p1(std::cos(a1) * Rb[k1], std::sin(a1) * Rb[k1]);

        // 切向 t 与法向 n（2D）
        glm::vec2 t2 = glm::normalize(p1 - p0);
        glm::vec2 n2(-t2.y, t2.x);

        // 内圈(i)和外圈(o) 3D 坐标（均在海平面上方一点）
        glm::vec3 i0(p0.x,              seaY + rimLift, p0.y);
        glm::vec3 i1(p1.x,              seaY + rimLift, p1.y);
        glm::vec3 o0(p0.x + n2.x*rimWidth, seaY + rimLift, p0.y + n2.y*rimWidth);
        glm::vec3 o1(p1.x + n2.x*rimWidth, seaY + rimLift, p1.y + n2.y*rimWidth);

        // 用几何构造法线：N = normalize( (i1-i0) × (o0-i0) )，朝上/外
        glm::vec3 N = glm::normalize(glm::cross(i1 - i0, o0 - i0));

        Vtx A{ i0,N,3.f,0 }, B{ i1,N,3.f,0 }, C{ o0,N,3.f,0 }, D{ o1,N,3.f,0 };
        addTri(m, A, C, B); // CCW
        addTri(m, B, C, D); // CCW
    }

    // ---------- (D) 水面 ----------
    // 简单双三角的大平面（略低于 seaY 以避免 Z 闪烁），kind=2
    {
        float W = worldR * 6.f;
        float wy = seaY - 0.04f; // 降低一点点，避免与 rim/岛底面共面
        glm::vec3 nW(0, 1, 0);
        glm::vec3 p0{ -W,wy,-W }, p1{ W,wy,-W }, p2{ -W,wy,W }, p3{ W,wy,W };
        Vtx A{ p0,nW,2.f,0 }, B{ p1,nW,2.f,0 }, C{ p2,nW,2.f,0 }, D{ p3,nW,2.f,0 };
        addTri(m, A, C, B); // CCW
        addTri(m, B, C, D); // CCW
    }

    // 上传到 GPU
    m.upload();
    return m;
}

// ====================== 6) 着色器（基础光照 + 雾） ======================

// 顶点着色器：把位置变换到裁剪空间，并输出世界空间坐标与法线、kind
static const char* VS = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in float aKind;
layout(location=3) in float aBand;

uniform mat4 uMVP, uModel; // 变换矩阵：MVP 与 Model
uniform mat3 uNrmMat;      // 法线矩阵（M 的逆转置的 3x3）

out vec3 vPosW;  // 世界空间位置
out vec3 vNrmW;  // 世界空间法线
out float vKind; // 片元类型

void main(){
    vec4 pw = uModel * vec4(aPos,1.0);
    vPosW = pw.xyz;
    vNrmW = normalize(uNrmMat * aNrm);
    vKind = aKind;
    gl_Position = uMVP * vec4(aPos,1.0);
}
)GLSL";

// 片元着色器：按 kind 选择基础色 + 漫反射，再与雾混合
static const char* FS = R"GLSL(
#version 330 core
in vec3 vPosW;
in vec3 vNrmW;
in float vKind;

uniform vec3 uCam;       // 摄像机位置（用于距离雾）
uniform vec3 uLightDir;  // 光方向（指向光源的反向，L = -dir）
uniform vec3 uHorizon;   // 地平线颜色（雾混合用）
uniform vec3 uSky;       // 天空颜色（雾混合用）
uniform float uFogNear;  // 雾起始距离
uniform float uFogFar;   // 雾结束距离

out vec4 FragColor;

// 颜色定义：可按需微调
vec3 landColor(){    return vec3(0.80,0.82,0.66); } // 陆地浅绿
vec3 plateauColor(){ return landColor() * 0.90; }   // 高原顶部略深（更易辨认）
vec3 waterColor(){   return vec3(0.50,0.68,0.84); } // 水面偏蓝
vec3 rimColor()  {   return vec3(0.97,0.97,0.99); } // 海岸/浪线近白
vec3 cliffColor(){   return vec3(0.97,0.97,0.99); } // 悬崖白

void main(){
    // 1) 根据 kind 选择基础色
    vec3 base = (vKind>3.5)? cliffColor() :
                (vKind>2.5)? rimColor()   :
                (vKind>1.5)? waterColor() :
                (vKind>0.5)? plateauColor():
                              landColor();

    // 2) 简单漫反射
    vec3 N = normalize(vNrmW);
    vec3 L = normalize(-uLightDir);
    float diff = max(dot(N,L), 0.0);

    // 固定的明暗范围：避免过对比，风格更卡通
    float kFlat = 0.85;                    // 基础亮度
    float kVar  = (vKind>1.5 ? 0.12 : 0.08); // 水/白边高一点变动，地面低一点
    vec3 col = base * (kFlat + kVar*diff);

    // 3) 线性雾：距离 uFogNear..uFogFar
    float d = length(uCam - vPosW);
    float f = clamp((d - uFogNear)/(uFogFar - uFogNear), 0.0, 1.0);
    vec3 fogCol = mix(uHorizon, uSky, 0.3); // 地平线到天空的过渡
    col = mix(col, fogCol, f);

    FragColor = vec4(col,1.0);
}
)GLSL";

// 编译/链接工具：失败打印信息并退出（便于快速定位 GLSL 报错）
static GLuint mkShader(GLenum t, const char* s) {
    GLuint sh = glCreateShader(t);
    glShaderSource(sh, 1, &s, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetShaderInfoLog(sh, len, nullptr, log.data());
        std::fprintf(stderr, "Shader error:\n%s\n", log.c_str());
        std::exit(EXIT_FAILURE);
    }
    return sh;
}
static GLuint mkProgram(const char* vs, const char* fs) {
    GLuint v = mkShader(GL_VERTEX_SHADER, vs);
    GLuint f = mkShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0'); glGetProgramInfoLog(p, len, nullptr, log.data());
        std::fprintf(stderr, "Link error:\n%s\n", log.c_str());
        std::exit(EXIT_FAILURE);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// ====================== 7) 简易自由摄像机与输入 ======================

struct Cam {
    // 初始位置/视角/FOV/移动速度/鼠标灵敏度
    glm::vec3 pos{ 0,3.0f,9.0f };
    float yaw   = -90.f;    // 左右
    float pitch = -15.f;    // 上下
    float fov   = 60.f;     // 视场角
    float speed = 6.f;      // 移动速度
    float sens  = 0.1f;     // 鼠标灵敏度

    bool first = true;      // 首帧标记（避免跳变）
    double lx = 0, ly = 0;  // 上一帧鼠标位置

    // 前向/右向/视图矩阵
    glm::vec3 fwd()const {
        float cy = std::cos(glm::radians(yaw)), sy = std::sin(glm::radians(yaw));
        float cp = std::cos(glm::radians(pitch)), sp = std::sin(glm::radians(pitch));
        return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
    }
    glm::vec3 right()const { return glm::normalize(glm::cross(fwd(), { 0,1,0 })); }
    glm::mat4 view()const  { return glm::lookAt(pos, pos + fwd(), { 0,1,0 }); }
};

// 全局输入状态
static bool gKeys[512]{}, gWire = false, gCap = true, gCull = true;

// 键盘回调
static void keyCB(GLFWwindow* w, int k, int, int a, int) {
    if (k >= 0 && k < 512) {
        if (a == GLFW_PRESS) gKeys[k] = true;
        else if (a == GLFW_RELEASE) gKeys[k] = false;
    }
    if (k == GLFW_KEY_ESCAPE && a == GLFW_PRESS) glfwSetWindowShouldClose(w, 1);
    if (k == GLFW_KEY_F1 && a == GLFW_PRESS)    gWire = !gWire;
    if (k == GLFW_KEY_F2 && a == GLFW_PRESS)    gCull = !gCull; // 开/关背面剔除
    if (k == GLFW_KEY_TAB && a == GLFW_PRESS) { // 捕获/释放鼠标
        gCap = !gCap;
        glfwSetInputMode(w, GLFW_CURSOR, gCap ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
}

// 鼠标回调（视角）
static void cursorCB(GLFWwindow* w, double x, double y) {
    Cam* c = (Cam*)glfwGetWindowUserPointer(w);
    if (!gCap) { c->first = true; return; } // 未捕获时不更新视角
    if (c->first) { c->lx = x; c->ly = y; c->first = false; } // 首帧对齐
    double dx = x - c->lx, dy = c->ly - y; // 注意 y 方向相反
    c->lx = x; c->ly = y;
    c->yaw   += (float)dx * c->sens;
    c->pitch += (float)dy * c->sens;
    c->pitch = glm::clamp(c->pitch, -89.f, 89.f); // 限制俯仰，避免万向锁
}

// ====================== 8) 主函数 ======================

int main() {
    // 初始化 GLFW 与 OpenGL 上下文
    Check(glfwInit() != 0, "glfwInit");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if _DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1280, 720,
        "Island + Plateaus with White Cliffs — CCW (cliff fixed, darker plateau)",
        nullptr, nullptr);
    Check(win != nullptr, "create window");
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1); // 开启 VSYNC

    // 加载 GL 函数
    Check(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) != 0, "glad");
    EnableGLDebugIfAvailable(); // Debug 模式尽量启用调试输出

    // 输入回调与摄像机
    glfwSetKeyCallback(win, keyCB);
    Cam cam; glfwSetWindowUserPointer(win, &cam);
    glfwSetCursorPosCallback(win, cursorCB);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // 默认捕获鼠标

    // 世界层级：海平面与陆地高度
    const float seaY  = 0.0f;
    const float landY = seaY + 0.10f; // 陆地稍高于海面

    // 生成岛屿网格（含：陆地+高原+海岸白边+水面）
    Mesh island = buildFlatIsland(
        /*worldR=*/6.5f,   // 世界尺度半径（越大越“广阔”）
        /*seg=*/  260,     // 角向分段（越大轮廓越平滑）
        /*seaY=*/ seaY,
        /*landY=*/landY,
        /*seed=*/ 1337.0f  // 随机种子
    );

    // 创建着色器程序 & 获取 uniform 位置
    GLuint prog = mkProgram(VS, FS);
    GLint uMVP  = glGetUniformLocation(prog, "uMVP");
    GLint uModel= glGetUniformLocation(prog, "uModel");
    GLint uNrm  = glGetUniformLocation(prog, "uNrmMat");
    GLint uCam  = glGetUniformLocation(prog, "uCam");
    GLint uLight= glGetUniformLocation(prog, "uLightDir");
    GLint uHor  = glGetUniformLocation(prog, "uHorizon");
    GLint uSky  = glGetUniformLocation(prog, "uSky");
    GLint uNear = glGetUniformLocation(prog, "uFogNear");
    GLint uFar  = glGetUniformLocation(prog, "uFogFar");

    // 基本渲染状态
    glEnable(GL_DEPTH_TEST);  // 深度测试
    glFrontFace(GL_CCW);      // CCW 为正面

    auto t0 = std::chrono::high_resolution_clock::now();
    double last = glfwGetTime();

    // 主循环
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        double now = glfwGetTime();
        float dt = (float)(now - last);
        last = now;

        // 背面剔除开关
        if (gCull) { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }
        else { glDisable(GL_CULL_FACE); }

        // 处理相机移动（WASD / Space / Ctrl）
        glm::vec3 f = cam.fwd(), r = cam.right(), up(0, 1, 0);
        float spd = cam.speed * (gKeys[GLFW_KEY_LEFT_SHIFT] ? 2.f : 1.f); // Shift 加速
        if (gKeys[GLFW_KEY_W])             cam.pos += f * spd * dt;
        if (gKeys[GLFW_KEY_S])             cam.pos -= f * spd * dt;
        if (gKeys[GLFW_KEY_A])             cam.pos -= r * spd * dt;
        if (gKeys[GLFW_KEY_D])             cam.pos += r * spd * dt;
        if (gKeys[GLFW_KEY_SPACE])         cam.pos += up * spd * dt;
        if (gKeys[GLFW_KEY_LEFT_CONTROL])  cam.pos -= up * spd * dt;

        // 让光缓慢环绕（仅为动态效果）
        float tt = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
        glm::vec3 lightDir = glm::normalize(glm::vec3(
            std::cos(tt * 0.1f) * 0.4f, // X
            -1.0f,                       // Y（向下）
            std::sin(tt * 0.1f) * 0.4f  // Z
        ));

        // 视口与清屏
        int W, H; glfwGetFramebufferSize(win, &W, &H);
        glViewport(0, 0, W, H);
        glClearColor(0.80f, 0.86f, 0.92f, 1.0f); // 天空背景色（与雾色一致）
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, gWire ? GL_LINE : GL_FILL);

        // 投影视图矩阵
        float aspect = H > 0 ? (float)W / H : 16.f / 9.f;
        glm::mat4 P = glm::perspective(glm::radians(cam.fov), aspect, 0.10f, 200.f);
        glm::mat4 V = cam.view();
        glm::mat4 M(1.f); // 世界变换（本例恒等）
        glm::mat4 MVP = P * V * M;
        glm::mat3 N   = glm::mat3(glm::transpose(glm::inverse(M))); // 法线矩阵

        // 绑定着色器与 uniform
        glUseProgram(prog);
        glUniformMatrix4fv(uMVP,  1, GL_FALSE, glm::value_ptr(MVP));
        glUniformMatrix4fv(uModel,1, GL_FALSE, glm::value_ptr(M));
        glUniformMatrix3fv(uNrm,  1, GL_FALSE, glm::value_ptr(N));
        glUniform3fv(uCam,   1, glm::value_ptr(cam.pos));
        glUniform3fv(uLight, 1, glm::value_ptr(lightDir));

        // 雾与天空/地平线颜色（可按需统一风格）
        glUniform3f(uHor, 0.76f, 0.84f, 0.90f); // 地平线
        glUniform3f(uSky, 0.80f, 0.86f, 0.92f); // 天空
        glUniform1f(uNear, 7.0f);  // 雾开始
        glUniform1f(uFar,  40.0f); // 雾结束

        // 绘制岛屿
        glBindVertexArray(island.vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)island.i.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glfwSwapBuffers(win);
    }

    // 清理
    island.destroy();
    glDeleteProgram(prog);
    glfwTerminate();
    return 0;
}
