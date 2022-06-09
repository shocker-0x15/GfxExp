#pragma once

#include "../../common_shader.h"

GLSL_UNIFORM(0, uvec2, u_screenSize);
GLSL_UNIFORM(1, vec3, u_cameraPosition);
GLSL_UNIFORM(2, mat4, u_prevMatW2C);
GLSL_UNIFORM(3, mat4, u_matW2C);
GLSL_UNIFORM(4, mat4, u_matW2CWithOffset);

GLSL_UNIFORM(5, mat4, u_prevMatM2W);
GLSL_UNIFORM(6, mat4, u_matM2W);
GLSL_UNIFORM(7, mat3, u_nMatM2W);
GLSL_UNIFORM(8, uint, u_instSlot);
GLSL_UNIFORM(9, uint, u_materialSlot);
GLSL_UNIFORM_BINDING(10, 0, sampler2D, u_normalTexture);
GLSL_UNIFORM(11, uint, u_flags);

GLSL_ATTRIBUTE(0, vec3, a_position);
GLSL_ATTRIBUTE(1, vec3, a_normal);
GLSL_ATTRIBUTE(2, vec3, a_tangent);
GLSL_ATTRIBUTE(3, vec2, a_texCoord);

#if _GLSL_FRAGMENT_SHADER_
layout(origin_upper_left) in vec4 gl_FragCoord;
#endif
GLSL_VARYING(vec3, v_positionInWorld);
GLSL_VARYING(vec3, v_tangentInWorld);
GLSL_VARYING(vec3, v_normalInWorld);
GLSL_VARYING(vec2, v_texCoord);
GLSL_VARYING(vec3, v_prevPositionInClip);
GLSL_VARYING(vec3, v_positionInClip);

GLSL_FRAGMENT_OUTPUT(0, vec4, gBuffer_positionInWorld_texCoord_x);
GLSL_FRAGMENT_OUTPUT(1, vec4, gBuffer_normalInWorld_texCoord_y);
GLSL_FRAGMENT_OUTPUT(2, vec4, gBuffer_prevScreenPos_instSlot_matSlot);
