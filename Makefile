.PHONY: clean

INCLUDES	:= -I/usr/local/cuda/include
LIBS		:= /usr/local/cuda/lib64/libcublas.so -ldl
CC		:= gcc
CFLAGS		+= -O2 -fPIC -fomit-frame-pointer $(INCLUDES)
LDFLAGS		+= -shared $(LIBS)

liboffloadcublas.so: liboffloadcublas.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

liboffloadcublas.o: liboffloadcublas.c gemm.c symm.c hemm.c syrk.c
	$(CC) $(CFLAGS) $< -c

clean:
	rm liboffloadcublas.so liboffloadcublas.o