#pragma once

extern int vdb_point(float x, float y, float z);
extern int vdb_line(float x0, float y0, float z0, float x1, float y1, float z1);
extern int vdb_normal(float x, float y, float z, float dx, float dy, float dz);
extern int vdb_triangle(float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2);

extern int vdb_color(float r, float g, float b);

extern int vdb_begin();
extern int vdb_end();

extern int vdb_frame();

extern int vdb_label(const char * lbl);
extern int vdb_label_i(int i);
