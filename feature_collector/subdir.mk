# Add inputs and outputs from these tool invocations to the build variables
CPP_SRCS += \
Environment.cpp \
FeatureCollector.cpp \
InfoCollector.cpp \
main.cpp 

OBJS += \
./Environment.o \
./FeatureCollector.o \
./InfoCollector.o \
./main.o 

CPP_DEPS += \
./Environment.d \
./FeatureCollector.d \
./InfoCollector.d \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ./%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


