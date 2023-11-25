#ifndef __VDB_H
#define __VDB_H

#ifndef _WIN32
#define VDB_CALL __attribute__((weak))
#define STRDUP strdup

#else
//#define VDB_CALL static
#define VDB_CALL

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#define snprintf sprintf_s
#define STRDUP _strdup

#endif

VDB_CALL int vdb_point(float x, float y, float z);
VDB_CALL int vdb_line(float x0, float y0, float z0, 
                      float x1, float y1, float z1);
VDB_CALL int vdb_normal(float x, float y, float z, 
                        float dx, float dy, float dz);
VDB_CALL int vdb_triangle(float x0, float y0, float z0, 
                          float x1, float y1, float z1,
                          float x2, float y2, float z2);

VDB_CALL int vdb_color(float r, float g, float b);

//By default, vdb will refresh the contents of the view at the end of every API call,
//you can surround multiple calls to vdb functions with vdb_begin/vdb_end
//to prevent the viewer from refreshing the contents until all the draw calls have been made
VDB_CALL int vdb_begin();
VDB_CALL int vdb_end();

//create a new blank frame. Currently just clears the screen, but eventually the viewer may keep
//around the contents of previous frames for inspection.
VDB_CALL int vdb_frame();

//associate a string label with the primitive being drawn
VDB_CALL int vdb_label(const char * lbl); 
VDB_CALL int vdb_label_i(int i); 

//for simplicity all the implementation to interface with vdb is in this header, just include it in your project

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
VDB_CALL int vdb_close(int i) { return close(i); }
#else
#include <WinSock2.h>
#pragma comment(lib, "Ws2_32.lib")
typedef int socklen_t;
VDB_CALL int vdb_close(int i) { return closesocket(i); }
#endif

static void vdb_report_error(const char * msg);

#include <stdio.h>
#include <stdlib.h>

#define VDB_BUFFER_SIZE (64*1024)
//maximum characters in a vdb command
#define VDB_REDZONE_SIZE 512



typedef struct {
	int is_initialized;
	int init_error; 
	int in_group;
	int fd;
	size_t n_bytes;
	char buffer[VDB_BUFFER_SIZE];
	
	char ** strings;
	int strings_N;
	int strings_allocated;
} __VDBState;

__VDBState __vdb;

static void vdb_os_init();

VDB_CALL void vdb_exit() {
	if(__vdb.init_error == 0) {
		vdb_close(__vdb.fd);
		__vdb.init_error = 1;
	}
}
VDB_CALL int vdb_init() {
	if(!__vdb.is_initialized) {
		__vdb.is_initialized = 1;
		vdb_os_init();
		__vdb.fd = socket(AF_INET, SOCK_STREAM, 0);
		if(__vdb.fd == -1) {
			vdb_report_error("");
			__vdb.init_error = 1;
		} else {
			struct sockaddr_in serv_name;
			serv_name.sin_family = AF_INET;
			serv_name.sin_addr.s_addr = htonl(0x7F000001L);
			serv_name.sin_port = htons(10000);
			if(-1 == connect(__vdb.fd, (struct sockaddr*) &serv_name, sizeof(serv_name))) {
				vdb_report_error("is the viewer open?");
				__vdb.init_error = 1;
			}
			atexit(vdb_exit);
		}
	}
	return __vdb.init_error;
}

#define VDB_INIT do { if(vdb_init()) return 1; } while(0)

VDB_CALL int vdb_flush() {
	unsigned int s;
	VDB_INIT;
	s = send(__vdb.fd,__vdb.buffer,__vdb.n_bytes,0);
	if(s != __vdb.n_bytes) {
		vdb_report_error("");
		__vdb.init_error = 1;
	}
	__vdb.n_bytes = 0;
	return 0;
}

VDB_CALL int vdb_printf(const char * fmt, ...) {
	va_list argp;
	VDB_INIT;
	va_start(argp,fmt);
	__vdb.n_bytes += vsnprintf(__vdb.buffer + __vdb.n_bytes, VDB_BUFFER_SIZE - __vdb.n_bytes,fmt,argp);
	va_end(argp);
	if(__vdb.in_group == 0 ||
	   (VDB_BUFFER_SIZE - __vdb.n_bytes < VDB_REDZONE_SIZE &&
	    __vdb.buffer[__vdb.n_bytes-1] == '\n')) {
		vdb_flush();
	}
	return 0;
}

VDB_CALL int vdb_begin() {
    __vdb.in_group++;
	return vdb_printf("b\n");
}
VDB_CALL int vdb_end() {
    if(__vdb.in_group > 0)
        __vdb.in_group--;
    return vdb_printf("e\n");
}
VDB_CALL int vdb_point(float x, float y, float z) {
	return vdb_printf("p %f %f %f\n",x,y,z);
}
VDB_CALL int vdb_line(float x0, float y0, float z0, float x1, float y1, float z1) {
	return vdb_printf("l %f %f %f %f %f %f\n",x0,y0,z0,x1,y1,z1);
}
VDB_CALL int vdb_normal(float x0, float y0, float z0, float x1, float y1, float z1) {
	return vdb_printf("n %f %f %f %f %f %f\n",x0,y0,z0,x1,y1,z1);
}

VDB_CALL int vdb_triangle(float x0, float y0, float z0, float x1, float y1, float z1,float x2, float y2, float z2) {
	return vdb_printf("t %f %f %f %f %f %f %f %f %f\n",x0,y0,z0,x1,y1,z1,x2,y2,z2);
}

VDB_CALL int vdb_frame() {
	vdb_printf("f\n");
	return vdb_flush();
}

VDB_CALL int vdb_color(float r, float g, float b) {
	return vdb_printf("c %f %f %f\n",r,g,b);
}

VDB_CALL int vdb_intern(const char * str) {
    int i,key;
    for(i = 0; i < __vdb.strings_N; i++)
        if(strcmp(__vdb.strings[i],str) == 0)
            return i;
    if(__vdb.strings_N == __vdb.strings_allocated) {
        if(__vdb.strings_allocated == 0)
            __vdb.strings_allocated = 16;
        else
            __vdb.strings_allocated *= 2;
        __vdb.strings = (char**) realloc(__vdb.strings,sizeof(char*)*__vdb.strings_allocated); 
    }
    __vdb.strings[__vdb.strings_N] = STRDUP(str);
    key = __vdb.strings_N++; 
    vdb_printf("s %d %s\n",key,str);
    return key;
}

VDB_CALL int vdb_label(const char * lbl) {
	int key;
	VDB_INIT;
	key = vdb_intern(lbl);
	vdb_printf("g %d\n",key);
	return 0;
}

VDB_CALL int vdb_label_i(int i) {
	char buf[128];
	snprintf(buf,128,"%d",i);
	return vdb_label(buf);
}

#ifdef _WIN32
static void vdb_report_error(const char * msg) {
    int errCode = WSAGetLastError();
    LPSTR errString = NULL;
    int size = FormatMessageA( FORMAT_MESSAGE_ALLOCATE_BUFFER |
                 FORMAT_MESSAGE_FROM_SYSTEM,
                 0,
                 errCode,
                 0,
                 (LPSTR)&errString,
                 0,          
                 0 );             
     fprintf(stderr, "vdb: %s %s (%d)\n", msg, errString, errCode);
     LocalFree( errString );
}
static void vdb_os_init() {
	WSADATA wsaData;
	if(WSAStartup(MAKEWORD(2,2), &wsaData)) {
    	exit(1);
    }
}
#else
static void vdb_os_init() {}
static void vdb_report_error(const char * msg) { fprintf(stderr,"vdb: %s %s\n",msg,strerror(errno)); }
#endif

#undef VDB_INIT
#undef VDB_CALL
#undef VDB_BUFFER_SIZE
#undef VDB_REDZONE_SIZE
#endif
