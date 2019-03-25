
CC=g++

CFLAGS=-std=c++11 -fPIC -Wall -Wextra -O3
BFLAGS=`python2.7-config --includes --libs` -lboost_python

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
# location of the Boost Python include files and library
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib

CacheSim.o: CacheSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c CacheSim.cpp

SNLRUSim.o: SNLRUSim.cpp CacheSim.o
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c SNLRUSim.cpp

ml: MLSim.o CacheSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o MLSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o MLSim.so
MLSim.o: MLSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c MLSim.cpp

gd: GDSim.o CacheSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o GDSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o GDSim.so
GDSim.o: GDSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c GDSim.cpp

lru: LRUSim.o CacheSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o LRUSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o LRUSim.so
LRUSim.o: LRUSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c LRUSim.cpp

s4lru: S4LRUSim.o SNLRUSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o SNLRUSim.o S4LRUSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o S4LRUSim.so
S4LRUSim.o: S4LRUSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c S4LRUSim.cpp

slru: SLRUSim.o SNLRUSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o SNLRUSim.o SLRUSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o SLRUSim.so
SLRUSim.o: SLRUSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c SLRUSim.cpp

adapt: AdaptSizeSim.o CacheSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o AdaptSizeSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o AdaptSizeSim.so
AdaptSizeSim.o: AdaptSizeSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c AdaptSizeSim.cpp

clean:
	rm *.so
	rm *.o

copy:
	cp *.so ../environment

all:
	make ml
	make gd
	make lru
	make s4lru
	make adapt
	make copy
