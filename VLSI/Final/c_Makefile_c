SOURCES += $(wildcard *.c)
OBJECTS := $(SOURCES:.c=.o)
CC = gcc
CFLAGS := -I.
CFLAGS += -Werror
LDFLAGS := -I.
EXECUTABLE := mips.elf

BUILD_DIR = build
LOG_DIR = logs

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c $(SOURCES)
	$(CC) -c $(CFLAGS) $< -o $@

run: $(EXECUTABLE)
	@mkdir -p $(LOG_DIR)
	./$(EXECUTABLE) > $(LOG_DIR)/run_log_$(shell date --iso=seconds).log

.PHONY: clean
clean:
	rm -f $(EXECUTABLE)
	rm -f *.o

echo: $(OBJECTS)
	@echo $(OBJECTS)