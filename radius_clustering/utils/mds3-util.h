/**
 * @file mds3-util.h
 * @brief Utility functions and data structures for MDS algorithms.
 *
 * This header file defines various utility functions, data structures,
 * and constants used in the implementation of MDS algorithms.
 * It includes helper functions for graph manipulation, random number generation,
 * and other common operations used across the MDS solver.
 */

#ifndef MDS3_UTIL_H
#define MDS3_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <sys/time.h>
#include <sys/resource.h>
#else
#error "Unsupported platform"
#endif

#define WORD_LENGTH 100
#define TRUE 1
#define FALSE 0
#define NONE 0
#define DELIMITER 0
#define PASSIVE 0
#define ACTIVE 1
#define MAX_NODE 10000000
#define max_expand_depth 100000
#define MAXIS 16

#define for_each_vertex(node) for(int node=1;node<=NB_NODE;node++)
#define for_each_neighbor(__vertex,__neibor)  for(int * __ptr=Node_Neibors[__vertex],__neibor=*__ptr;__neibor!=NONE;__neibor=*(++__ptr))

#define domed(node) (STATUS[node].dominated)
#define clr_domed_status(node) (STATUS[node].dominated=0)
#define set_domed_status(node) (STATUS[node].dominated=1)

#define fixed(node) (STATUS[node].fixed)
#define deleted(node) (STATUS[node].deleted)

#define removed(node) (STATUS[node].removed)
#define set_removed_status(node) (STATUS[node].removed=1)
#define clr_removed_status(node) (STATUS[node].removed=0)

#define branched(node) (STATUS[node].branched)
#define set_branched_status(node) (STATUS[node].branched=1)
#define clr_branched_status(node) (STATUS[node].branched=0)

#define active(node) (STATUS[node].active)
#define set_active(node) (STATUS[node].active=1)
#define clr_active(node) (STATUS[node].active=0)

#define included(node)  (STATUS[node].included)
#define set_included_status(node)  (STATUS[node].included=1)
#define clr_included_status(node)  (STATUS[node].included=0)

#define bit_set(vec,idx) ((*(vec+(idx>>5)))|= (1<<(idx&31)))
#define bit_clr(vec,idx) ((*(vec+(idx>>5)))&= (~(1<<(idx&31))))
#define bit_val(vec,idx) ((*(vec+(idx>>5)))&(1<<(idx&31)))
#define CUR_BRA_IDX BRAIDX[CUR_LEVEL]
#define CUR_BRA_NODE  ITEM(BRA_STK,CUR_BRA_IDX)
#define CUR_LEVEL_UND_IDX  UNDIDX[CUR_LEVEL]
#define adjlen(node) ((node)/32+1)

#define marked(node) (PID[node].marked)
#define set_marked_status(node) (PID[node].marked=1)
#define clr_marked_status(node) (PID[node].marked=0)
  
#define involved(node) (PID[node].involved)
#define set_involved_status(node) (PID[node].involved=1)
#define clr_involved_status(node) (PID[node].involved=0)
#define set_newid(node,id)  (PID[node].newid=id)
#define set_isno(node,no)  (PID[node].isno=no)


#define newid(node)  (PID[node].newid)
#define isno(node)  (PID[node].isno)

#define branch_node_at_level(i) ITEM(BRA_STK,BRAIDX[i])

// Macro for vector  
#define VEC_DECLARE(T,tName) \
typedef struct { \
	T *addr; \
    unsigned used;\
	unsigned capacity;\
}tName

#define push_back(Vec,T,Val)  \
do{ \
	assert(Vec->used<=Vec->capacity);\
    if(Vec->used==Vec->capacity){ \
		int size=Vec->capacity*2; \
		Vec->addr=(T *)realloc(Vec->addr,(size+1)*sizeof(T));\
        assert(Vec->addr!=NULL);  \
		Vec->capacity=size;\
	}\
	*(Vec->addr+Vec->used)=(Val);\
	(Vec->used)++;\
}while(0)

#define new_stack(Vec,VEC_TYPE,ITEM_TYPE,len) \
do{ \
assert(len>0);\
unsigned size=(len);\
Vec=(VEC_TYPE *)calloc(1,sizeof(VEC_TYPE));\
assert(Vec!=NULL);\
(Vec)->addr=(ITEM_TYPE *)malloc((size+1)*sizeof(ITEM_TYPE)); \
assert((Vec)->addr); \
(Vec)->capacity=size;\
(Vec)->used=0;\
}while(0)

#define free_stack(Vec) \
do{ \
if(Vec!=NULL){\
free(Vec->addr);\
free(Vec);\
}\
}while(0)

#define for_each_vec_item(Vec,T,It) for(T *It=Vec->addr, *__end=Vec->addr+Vec->used;It != __end;It++)

#define remove_value_from_vector(Vec,T,Val) do{				\
for(T *It=Vec->addr, *__end=Vec->addr+Vec->used;It != __end;)\
if(*It==Val){\
Vec->used--;*It=*(Vec->addr+Vec->used);\
__end--;\
}else \
  It++;\
}while(0)

#define ITEM(VEC,IDX) (VEC->addr[(IDX)])
#define USED(VEC) (VEC->used)

VEC_DECLARE(int,VEC_INT);
VEC_DECLARE(unsigned,VEC_UINT);

/*  end of vector  */
/*
typedef struct{
  unsigned fixed:1;
  unsigned active:1;
  unsigned deleted:1;
  unsigned removed:1;
  unsigned included:1;
  unsigned branched:1;
  unsigned dominated:1;
  unsigned undidx:25;
}VSTATUS;
*/
typedef struct{
  char fixed:1;
  char active:1;
  char deleted:1;
  char removed:1;
  char included:1;
  char branched:1;
  char dominated:1;
  char future:1;
}VSTATUS;

typedef struct{
 unsigned involved:1;
 unsigned marked:1;
 unsigned newid:20;
 unsigned isno:10;
}PSTATUS;


static int * Init_Adj_List;
static int BLOCK_COUNT = 0;
static int *BLOCK_LIST[100];
static int **Node_Neibors;

static unsigned Node_Degree[MAX_NODE];

static int NB_NODE,NB_EDGE, Max_Degree = 0, Max_Degree_Node,SUB_PROBLEM_SIZE;
static int FORMAT = 1,  NB_NODE_O,  NB_EDGE_O;
static double READ_TIME, INIT_TIME, SEARCH_TIME;
static double D0 = 0, D1 = 0, D2 = 0, Dt = 0;
static int INIT_BRANCHING_NODE=0,INIT_UPPER_BOUND=0;
static unsigned long long NB_TREE=0;

static int * CFG;
static int * LOC;
static int * BRAIDX;
static int * UNDIDX;
static VEC_INT * BRA_STK;
static int   BEST_LEVEL,CUR_LEVEL,CUR_UND_IDX;
static VSTATUS * STATUS;
static PSTATUS * PID;
static int * ADJIDX;
static VEC_UINT * ADJ_STK;
static int TIME_OUT, CUT_OFF=0;
static double BEST_SOL_TIME;
static char instance[1024]={'\0'};
static VEC_INT *iSET[MAXIS+1];
static int iSET_Counter[MAXIS+1];
static int iSET_Status[MAXIS];
static float *Node_Score;

struct Result {
  int* dominating_set;
  int set_size;
  double exec_time;
};

static double get_utime() {
  #ifdef _WIN32
      FILETIME createTime;
      FILETIME exitTime;
      FILETIME kernelTime;
      FILETIME userTime;
      if (GetProcessTimes(GetCurrentProcess(),
                          &createTime, &exitTime,
                          &kernelTime, &userTime) != 0) {
         ULARGE_INTEGER li = {{userTime.dwLowDateTime, userTime.dwHighDateTime}};
         return li.QuadPart * 1e-7;
      }
        return 0.0;
  #elif defined(__APPLE__) || defined(__linux__)
    struct rusage utime;
    if (getrusage(RUSAGE_SELF, &utime) == 0) {
      return (double)utime.ru_utime.tv_sec + (double)utime.ru_utime.tv_usec * 1e-6;
    }
    return 0.0;
  #else
    return (double)clock() / CLOCKS_PER_SEC;
  #endif
}

static int cmp_branching_vertex_score(const void * a, const void *b){
  return Node_Degree[*((int *) b)] - Node_Degree[*((int *) a)];
}

static int int_cmp_desc(const void * a, const void * b) {
  return *((int *) b) - *((int *) a);
}

static int int_cmp_asc(const void * a, const void * b) {
  return *((int *) a) - *((int *) b);
}
static VEC_INT * FIX_STK,* TMP_STK;
static VEC_INT * VEC_SUBGRAPHS;
static VEC_INT * VEC_SOLUTION;
static int NB_FIXED=0,NEW_IDX=0,NB_UNFIXED=0;


static void allcoate_memory_for_adjacency_list(int nb_node, int nb_edge,int offset) {
  int i, block_size = 40960000;
  unsigned int free_size = 0;
  Init_Adj_List = (int *) malloc((2 * nb_edge + nb_node) * sizeof(int));
  if (Init_Adj_List == NULL ) {
    for (i = 1; i <= NB_NODE; i++) {
      if (Node_Degree[i - offset] + 1 > free_size) {
	Node_Neibors[i] = (int *) malloc(block_size * sizeof(int));
	BLOCK_LIST[BLOCK_COUNT++] = Node_Neibors[i];
	free_size = block_size - (Node_Degree[i - offset] + 1);
      } else {
	Node_Neibors[i] = Node_Neibors[i - 1]
	  + Node_Degree[i - 1 - offset] + 1;
	free_size = free_size - (Node_Degree[i - offset] + 1);
      }
    }
  } else {
    BLOCK_COUNT = 1;
    BLOCK_LIST[BLOCK_COUNT - 1] = Init_Adj_List;
    Node_Neibors[1] = Init_Adj_List;
    for (i = 2; i <= NB_NODE; i++) {
      Node_Neibors[i] = Node_Neibors[i - 1] + Node_Degree[i - 1 - offset]
	+ 1;
    }
  }
}


static int _read_graph_from_edge_list(unsigned int* edges, int n, int nb_edges) {
  int i, j, l_node, r_node, nb_edge = 0, max_node = n, offset = 0;
  int node = 1;

  memset(Node_Degree, 0, (MAX_NODE) * sizeof(int));
  
  for (j =0; j < 2 * nb_edges; j+=2) {
    l_node = edges[j];
    r_node = edges[j+1];

    if (l_node >= 0 && r_node >= 0 && l_node != r_node) {
      nb_edge++;
      if (l_node > max_node) max_node = l_node;
      if (r_node > max_node) max_node = r_node;
      
    }
    if (offset ==0 && (l_node == 0 || r_node == 0)){
      offset = 1;
    }
    Node_Degree[l_node]++;
    Node_Degree[r_node]++;
  }
  NB_NODE = max_node;

  Node_Neibors = (int **)malloc((NB_NODE + 1) * sizeof(int *));
  allcoate_memory_for_adjacency_list(NB_NODE, nb_edge, 1);
  memset(Node_Degree, 0, (NB_NODE + 1) * sizeof(int));

  nb_edge = 0;
  for (j = 0; j < 2 * nb_edges; j+=2) {
    l_node = edges[j];
    r_node = edges[j+1];
    if (l_node >= 0 && r_node >= 0 && l_node != r_node) {
      if (offset) {
        l_node += offset;
        r_node += offset;
      }
      for (i = 0; i < Node_Degree[l_node]; i++) {
        if (Node_Neibors[l_node][i] == r_node) 
          break;
        
      }
      if (i == Node_Degree[l_node]) {
        Node_Neibors[l_node][Node_Degree[l_node]] = r_node;
        Node_Neibors[r_node][Node_Degree[r_node]] = l_node;
        Node_Degree[l_node]++;
        Node_Degree[r_node]++;
        nb_edge++;
      }
      
    }
  }
  NB_EDGE = nb_edge;
  Max_Degree = 0;
  for (node = 1; node <= NB_NODE; node++) {
    Node_Neibors[node][Node_Degree[node]] = NONE;
    if (Node_Degree[node] > Max_Degree) {
      Max_Degree = Node_Degree[node];
      Max_Degree_Node = node;
    }
  }
  return TRUE;
}
   
static void free_block() {
  int i = 0;
  for (i = 0; i < BLOCK_COUNT; i++)
    free(BLOCK_LIST[i]);
}

static int *dfn,*low,*TarStack,TarTop,CNT=0,*SonNum,*RecSta,RecTop,*LasSon,*LasNodeIndex;

//After preprocess, following variables might still be useful:
static int *SubGraph_size,NB_DCC=0,*InDcc,NB_cut=0;
static double REDUCE_TIME=0;
//NB_DCC indicates the number of v-DCCs(sub-graphs)
//NB_cut indicates the number of cut-vertexes
//SubGraph_size,0-indexed,"SubGraph_size[i]=j" indicates that there are j nodes in the i-th subgraph;
//Cut[x]=1 indicates that x is a cut-vertex
//InDcc[x]=y indicates that vertex x is in the y-th v-DCC; especially, if Cut[x]==1, InDcc[x]=-1, since a cut-vertex might be involved by several v-DCCs



static inline void partition_oneproblem(){
  #ifdef NOR
  //	Branching_Queue=(int *)malloc(sizeof(int)*(NB_NODE+1));
	NB_UNFIXED=NB_NODE;
  #endif
  new_stack(VEC_SUBGRAPHS,VEC_INT,int,NB_NODE);
  for(int i=1;i<=NB_NODE;++i){
    if(!deleted(i)){
      push_back(VEC_SUBGRAPHS,int,i); 
    }
  }
  push_back(VEC_SUBGRAPHS,int,NONE);
}

static inline void reduce_graph2(){
  
  for(int node=1;node<=NB_NODE;node++){
    if(deleted(node))continue;

    set_marked_status(node);
    for_each_neighbor(node,neibor){
      set_marked_status(neibor);
    }

    for_each_neighbor(node,neibor){
      if(fixed(neibor)){
	set_branched_status(neibor);
	continue;
      }
      for_each_neighbor(neibor,neibor2){
	if(!marked(neibor2)){
	  set_branched_status(neibor);
	  break;
	}
      }
    }

    for_each_neighbor(node,neibor){
      if(branched(neibor))
	continue;
      for_each_neighbor(neibor,neibor2){
	if(branched(neibor2)){
	  set_involved_status(neibor);
	  break;
	}
      }
      if(!involved(neibor)){
	fixed(node)=1;
	break;
      }
    }
    
    if(fixed(node)){
      for_each_neighbor(node,neibor){
	if(!branched(neibor))
	  deleted(neibor)=1;
      }
    }

    clr_marked_status(node);
    for_each_neighbor(node,neibor){
      clr_marked_status(neibor);
      clr_involved_status(neibor); 
      clr_branched_status(neibor);
    }
  }

  //reduce adjlist
  for(int node=1;node<=NB_NODE;node++){
    if(fixed(node))
      NB_FIXED++;
    if(deleted(node))
      continue;
    
    int *ptr=Node_Neibors[node],count=0;
    for_each_neighbor(node,neibor){
      if(!deleted(neibor)){
	*ptr++=neibor;count++;
      }
    }
    *ptr=NONE;
    Node_Degree[node]=count;
  }
}

static inline void reduce_graph(){
	int *Que,*Col,*Dis,*InQue,Ql,Qr;
	int *Fixed,*Deleted;
	Que=(int *)malloc((NB_NODE+1)*sizeof (int));
	Fixed=(int *) malloc((NB_NODE+1)*sizeof(int));
	Deleted=(int *) malloc((NB_NODE + 1) * sizeof(int));
	Col=(int *) malloc((NB_NODE+1)*sizeof (int));
	Dis=(int *) malloc((NB_NODE+1)*sizeof (int));
	InQue=(int *) malloc((NB_NODE+1)*sizeof (int));
	memset(Fixed,0,(NB_NODE+1)*sizeof (int));
	memset(Deleted, 0, (NB_NODE + 1) * sizeof (int));
	memset(Col,0,(NB_NODE+1)*sizeof (int));
	memset(InQue,0,(NB_NODE+1)*sizeof (int));
	memset(Dis,0x3f,(NB_NODE+1)*sizeof (int));
	const int INF=0x3f3f3f3f;
	for(int i=1;i<=NB_NODE;i++){
		if(Deleted[i])continue;
		Ql=0;Qr=0;
		Que[Qr++]=i;Dis[i]=0;InQue[i]=1;
		int Me,tt,ColCnt=0;
		while(Ql!=Qr) {
			Me = Que[Ql++];
			if (Dis[Me] == 2)break;
			for (int j = 0; j < Node_Degree[Me]; j++) {
				tt = Node_Neibors[Me][j];
				//if(Deleted[tt])continue;
				//if(Me==1)printf("**%d\n",tt);
				if (Dis[tt] <= 1)continue;
				if(Dis[tt]>Dis[Me]+1)Dis[tt] = Dis[Me] + 1;
				if(!InQue[tt]){
					Que[Qr++] = tt;
					InQue[tt]=1;
				}
				if (Dis[tt] == 2 && Dis[Me]==1 && Col[Me] == 0) {
					Col[Me] = 1;
					ColCnt++;
				}
			}
		}
		int NeiNum=0;
		for(int j=1;j<Qr;j++){
			if(Dis[Que[j]]==2)break;
			NeiNum++;
			if(Col[Que[j]]==1)continue;
			Me=Que[j];
			for(int k=0;k<Node_Degree[Me];k++){
				tt=Node_Neibors[Me][k];
				//if(Deleted[tt])continue;
				if(Col[tt]==1){
					ColCnt++;
					Col[Me]=2;
					break;
				}
			}
		}
		if(ColCnt<NeiNum){
			Fixed[i]=1;
			//printf("Fixed! *%d(%d,%d)\n",i,ColCnt,NeiNum);
			for(int j=1;j<Qr;j++){
				tt=Que[j];
				if(Dis[tt]==2)break;
				if(Col[tt]==1)continue;
				Deleted[tt]=1;
			}
		}
		for (int j=0;j<Qr;++j) {
			Me=Que[j];
			Dis[Me]=INF;
			Col[Me]=0;
			InQue[Me]=0;
		}
	}
	NB_UNFIXED=0;
	for(int i=1;i<=NB_NODE;i++){
	  if(Deleted[i]){
	    deleted(i)=1;
	    continue;
	  }
	  if(Fixed[i]){
	    fixed(i)=1;
	    NB_FIXED++;
	  }else{
	    NB_UNFIXED++;
	  }
	
	  int NeiCnt=0;
	  for(int j=0;j<Node_Degree[i];j++){
	    int tt=Node_Neibors[i][j];
	    if(Deleted[tt])continue;
	    Node_Neibors[i][NeiCnt++]=tt;
	  }
	  Node_Degree[i]=NeiCnt;
	  Node_Neibors[i][NeiCnt]=NONE;
	}
	//printf("\n");
	free(Que);
	free(Col);
	free(Dis);
	free(InQue);
	free(Fixed);
	free(Deleted);
    
	REDUCE_TIME = get_utime()-READ_TIME;

	fflush(stdout);
}

// Declare functions as extern
extern void check_final_solution();
extern void update_best_solution();
extern void check_and_save_solution();
extern void check_consistance();
extern void cleanup();
extern int max_dominated_number();
extern int select_branching_node();
extern void search_domset();
extern int fast_search_initial_solution();
extern void solve_subproblems();
extern struct Result* emos_main(unsigned int* edges, int n, int nb_edge);
extern int* get_dominating_set(struct Result* result);
extern int get_set_size(struct Result* result);
extern double get_exec_time(struct Result* result);
extern void free_results(struct Result* result);

// Declare global variables as extern
extern unsigned long long total_branches;
extern unsigned long long pruned_branches;

#endif // MDS3_UTIL_H