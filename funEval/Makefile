SRC = main.c gpuSetup.c
HDR = gpuSetup.h
KRN = funEvalKernel.cl

BIN = feval
LIB = 

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIB += -framework OpenCL
else
	LIB += -l OpenCL
endif

FLG = -Wall -O3  # -DUSE_DOUBLE

default: $(BIN)

$(BIN): $(SRC) $(HDR) $(KRN)
	gcc $(FLG) -o $@ $(SRC) $(LIB)

clean:
	rm -f $(BIN)
