SRC = main.c gpusetup.c
HDR = gpusetup.h
KRN = funEvalKernel.cl

BIN = feval

LIB = -framework OpenCL
FLG = 

default: $(BIN)

$(BIN): $(SRC) $(HDR) $(KRN)
	gcc $(FLG) -o $@ $(SRC) $(LIB)

clean:
	rm -f $(BIN)
