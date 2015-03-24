# C compiler
# ----------
CC            = gcc
CFLAGS        = -fPIC -Wall -O3 -std=c99

# Compiler and linker options
# ---------------------------
LIBFLAGS      =
INCFLAGS      =
LOCAL_LDFLAGS =

# Project files
# -------------
SONAME = lib3x3.so.1
BIN    = $(SONAME).0.0
OBJ    = slvsec3.o dsyev2.o dsytrd3.o dsyevc3.o dsyevj3.o dsyevq3.o dsyevv3.o dsyevh3.o \
	 dsyevd3.o zhetrd3.o zheevc3.o zheevj3.o zheevq3.o zheevv3.o zheevh3.o zheevd3.o
HEADER = slvsec3.h dsyev2.h dsytrd3.h dsyevc3.h dsyevj3.h dsyevq3.h dsyevv3.h dsyevh3.h \
	 dsyevd3.h zhetrd3.h zheevc3.h zheevj3.h zheevq3.h zheevv3.h zheevh3.h zheevd3.h


# Rules
# -----
$(BIN): Makefile $(OBJ)
	$(CC) -shared -Wl,-soname,$(SONAME) $(CFLAGS) $(INCFLAGS) $(OBJ) $(LIBFLAGS) \
              $(LDFLAGS) $(LOCAL_LDFLAGS) -o $(BIN)

%.o : %.c Makefile $(HEADER)
	$(CC) $(CFLAGS) -c $(INCFLAGS) $<

.PHONY: clean
clean:
	rm -f $(BIN) $(OBJ)


