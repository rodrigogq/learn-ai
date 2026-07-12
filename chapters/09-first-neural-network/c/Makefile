# Builds every .c file in this folder into build/<name>.
# Used identically in every chapter: `make` builds, `make clean` removes build/.

CC = cc
CFLAGS = -std=c11 -Wall -Wextra -O2
LDLIBS = -lm

SOURCES := $(wildcard *.c)
BINARIES := $(SOURCES:%.c=build/%)

all: $(BINARIES)

build/%: %.c
	@mkdir -p build
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -rf build

.PHONY: all clean
