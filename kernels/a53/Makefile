CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror

ifeq ($(shell uname -m),aarch64)
CFLAGS += -march=armv8-a -mtune=cortex-a53
endif

.PHONY: all check clean
all: ternary_gemv

ternary_gemv: ternary_gemv.c
	$(CC) $(CFLAGS) $< -o $@

check: ternary_gemv
	./ternary_gemv 128 64 2

clean:
	$(RM) ternary_gemv
