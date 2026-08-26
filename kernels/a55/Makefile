CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror

ifeq ($(shell uname -m),aarch64)
CFLAGS += -march=armv8.2-a+dotprod -mtune=cortex-a55
endif

.PHONY: all check clean assembly
all: ternary_dotprod

ternary_dotprod: ternary_dotprod.c ../a53/ternary_gemv.c
	$(CC) $(CFLAGS) ternary_dotprod.c -o $@

check: ternary_dotprod
	./ternary_dotprod 128 64 2
	./ternary_dotprod --nibble 128 64 2

assembly:
	$(CC) $(CFLAGS) -S ternary_dotprod.c -o ternary_dotprod.s

clean:
	$(RM) ternary_dotprod ternary_dotprod.s
