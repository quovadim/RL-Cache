CC=g++

CFLAGS=-std=c++11 -fPIC -Wall -Wextra -O3
BFLAGS=`python2.7-config --includes --libs` -lboost_python

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
# location of the Boost Python include files and library
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib

interface: CacheSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o CacheSim.so
CacheSim.o: CacheSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c CacheSim.cpp

ml: CacheMLSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o CacheMLSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o CacheMLSim.so
CacheMLSim.o: CacheMLSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c CacheMLSim.cpp

adm: CacheMLSimAdm.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o CacheMLSimAdm.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o CacheMLSimAdm.so
CacheMLSimAdm.o: CacheMLSimAdm.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c CacheMLSimAdm.cpp

mllru: CacheMLSimAdmLRU.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o CacheMLSimAdmLRU.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o CacheMLSimAdmLRU.so
CacheMLSimAdmLRU.o: CacheMLSimAdmLRU.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c CacheMLSimAdmLRU.cpp

lru: LRUSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o LRUSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o LRUSim.so
LRUSim.o: LRUSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c LRUSim.cpp

gdsf: GDSFSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o GDSFSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o GDSFSim.so
GDSFSim.o: GDSFSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS) -c GDSFSim.cpp

s4lru: S4LRUSim.o
	g++ -shared $(CFLAGS),--export-dynamic CacheSim.o S4LRUSim.o -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o S4LRUSim.so
S4LRUSim.o: S4LRUSim.cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(CFLAGS)-c S4LRUSim.cpp

clean:
	rm *.so
	rm *.o

copy:
	cp *.so ../

all:
	make interface
	make ml
	make adm
	make mllru
	make lru
	make gdsf
	make copy

