#pragma once

#if _GLSL_
#   define GLSL_UNIFORM(Location, Type, Name) layout(location = Location) uniform Type Name
#   define GLSL_UNIFORM_BINDING(Location, Binding, Type, Name) layout(location = Location, binding = Binding) uniform Type Name
#else
#   define GLSL_UNIFORM(Location, Type, Name) static Type Name
#   define GLSL_UNIFORM_BINDING(Location, Binding, Type, Name) static Type Name
#endif

#if _GLSL_VERTEX_SHADER_
#   define GLSL_ATTRIBUTE(Location, Type, Name) layout(location = Location) in Type Name
#   define GLSL_VARYING(Type, Name) out Type Name
#   define GLSL_FRAGMENT_OUTPUT(Location, Type, Name)
#elif _GLSL_FRAGMENT_SHADER_
#   define GLSL_ATTRIBUTE(Location, Type, Name)
#   define GLSL_VARYING(Type, Name) in Type Name
#   define GLSL_FRAGMENT_OUTPUT(Location, Type, Name) layout(location = Location) out Type Name
#endif
