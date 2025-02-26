/*
EMOS algorithm for solving minimum dominating set problem.
Copyright (C) 2023, Hua Jiang.

Adaptation of the algorithm to python interface.
Copyright (C) 2024, Haenn Quentin.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
    #include <direct.h>
    #define SIGINT 2
    typedef void (*SignalHandlerFn)(int);
#elif defined(__APPLE__) || defined(__linux__)
    #include <sys/time.h>
    #include <sys/resource.h>
    #include <sys/types.h>
    #include <unistd.h>
    #include <signal.h>
#else
    #error "Unsupported platform"
#endif


#include "mds3-util.h"
#include "util_heap.h"



static inline void swap_cfg(int a, int b){
  CFG[LOC[b]]=a;
  CFG[LOC[a]]=b;
  int t=LOC[a];
  LOC[a]=LOC[b];
  LOC[b]=t;
}

static inline void rollback_branch_node(int bnode){
  for(int i=CUR_LEVEL_UND_IDX;i<CUR_UND_IDX;i++){
    clr_domed_status(CFG[i]);
  }
  CUR_UND_IDX=CUR_LEVEL_UND_IDX;
  if(!CUR_UND_IDX){
    clr_domed_status(bnode);
  }
}

static inline void backtrack(){
  CUR_LEVEL--;
  int i=USED(BRA_STK)-2,node;
  while((node=ITEM(BRA_STK,i))!=NONE){
    clr_branched_status(node);
    i--;
  }
  USED(BRA_STK)= i+1;
}

static void initialize(){
  //allocate memory
  CFG=(int *)malloc((NB_NODE+1)*sizeof(int));
  LOC=(int *)malloc((NB_NODE+1)*sizeof(int));
  // BEST_DOMSET=(int *)malloc((NB_NODE+1)*sizeof(int));
  STATUS=(VSTATUS *)calloc((NB_NODE+1),sizeof(VSTATUS));
 
  PID=(PSTATUS *)calloc((NB_NODE+1),sizeof(PSTATUS));

  //initialize stack
  new_stack(BRA_STK,VEC_INT,int,NB_NODE);
  new_stack(VEC_SOLUTION,VEC_INT,int,256);
  new_stack(FIX_STK,VEC_INT,int,256);
  new_stack(TMP_STK,VEC_INT,int,128);
  new_stack(ADJ_STK,VEC_UINT,unsigned,1);
  
  for(int i=0;i<=MAXIS;i++){
    new_stack(iSET[i],VEC_INT,int,64);
  }

  
  
  #ifdef ADJ
  ADJIDX=(int *)malloc((NB_NODE+1)*sizeof(int));
  ADJIDX[1]=0;
  for(int n=2;n<=NB_NODE;n++){
    ADJIDX[n]=ADJIDX[n-1]+adjlen(n-1);
  }
  #endif
  memset(STATUS,0,(NB_NODE+1)*sizeof(VSTATUS));
  memset(PID,0,(NB_NODE+1)*sizeof(PSTATUS));

  #ifdef CHECK
   for(int node=1;node<=NB_NODE;node++){
     assert(fixed(node)==0);
     assert(active(node)==0);
     assert(deleted(node)==0);
     assert(removed(node)==0);
     assert(included(node)==0); 
     assert(branched(node)==0);
     assert(domed(node)==0);
  }
  #endif
}


static inline int is_2adj(int n1,int n2){
  if(n1>n2){
    unsigned int * vec=ADJ_STK->addr+ADJIDX[n1];
    return  bit_val(vec,n2)? 1:0;
  }else{
    unsigned int * vec=ADJ_STK->addr+ADJIDX[n2];
    return  bit_val(vec,n1)? 1:0;
  }  
}

static inline void set_adj_bit(int n1,int n2){
  unsigned int * vec=ADJ_STK->addr+ADJIDX[n1];
  bit_set(vec,n2);
}

static int test_2adj(int node1,int node2){
  int flag=0;
  for_each_vertex(node){
    assert(marked(node)==0);
  }
  set_marked_status(node1);
  for_each_neighbor(node1,neibor){
    if(involved(neibor)){
      set_marked_status(neibor); 
    }
  }
  int adj=0;
  for_each_neighbor(node2,neibor){
    if(neibor==node1)
      adj=1;
    if(involved(neibor) && !branched(neibor) && marked(neibor)){
      flag=1;
      break;
    }
  }
  clr_marked_status(node1);
  for_each_neighbor(node1,neibor){
    if(involved(neibor))
      clr_marked_status(neibor);
  }
  if(flag)
  return flag;
  if(adj)
    return branched(node2)? 0:1;
  else
    return 0;
}

static void test_adjacent(){
   for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
     if(!involved(CFG[i]))continue;
     int n1=newid(CFG[i]);
     
     for(int j=i+1;j<SUB_PROBLEM_SIZE;j++){
       if(involved(CFG[j])){
	 int n2=newid(CFG[j]);
	 if(is_2adj(n1,n2)){
           if(!test_2adj(CFG[i],CFG[j])){
	     printf("%d(%d) %d(%d)\n",CFG[i],n1,CFG[j],n2);
	   }
	   assert(test_2adj(CFG[i],CFG[j])==1);
	 }
	 else{
	   if(test_2adj(CFG[i],CFG[j])){
	     printf("%d(%d) %d(%d)\n",CFG[i],n1,CFG[j],n2);
	   }
	   assert(test_2adj(CFG[i],CFG[j])==0);
	 }
       }
     }
   }
}
/*
this function is used to determine if two vertices are 2-adj
precondition: all neighbors of marked_node are marked.
two vertices adj iif:
1) they two are adj and at least one isn't branched.
2) they have a common neighbor of unbranched. 
*/

static inline  int is_2_adj(int marked_node,int test_node){
   for_each_neighbor(test_node,neibor){
     if(neibor==marked_node){
       if(!(branched(test_node) && branched(marked_node)))
	 return 1;
     }else if(involved(neibor) && !branched(neibor) && marked(neibor))
       return 1;
   }
   return 0;
}

static void update_adj_matrix(int node,int node_idx,int new_idx){
  for(int i=0;i<adjlen(new_idx);i++)
    push_back(ADJ_STK,unsigned,0);

  for(int j=node_idx+1;j<SUB_PROBLEM_SIZE;j++){
    int node1=CFG[j];
    
    if(!involved(node1))continue;

    if(is_2_adj(node,node1))
       set_adj_bit(new_idx,PID[node1].newid);

    if(!branched(node) && marked(node1)){
      for(int k=j+1;k<SUB_PROBLEM_SIZE;k++){
	  int node2=CFG[k];
	  if(involved(node2) && marked(node2))
	    set_adj_bit(PID[node1].newid,PID[node2].newid);
      }
    }
  }
}

static void build_adjacent_matrix_for_free_nodes(){
  USED(ADJ_STK)=0;
  for(int i=SUB_PROBLEM_SIZE-1,idx=1;i>=CUR_UND_IDX;i--,idx++){
    int node=CFG[i];
    // printf(" building %d \n",node);
    assert(!domed(node));
    set_newid(node,idx);
    set_involved_status(node);
    set_marked_status(node);
    for(int j=0;j<adjlen(idx);j++)
      push_back(ADJ_STK,unsigned,0);
    
    for_each_neighbor(node,neibor){
      if(active(neibor) && involved(neibor))
	set_marked_status(neibor);
    }
    
    for(int j=i+1;j<SUB_PROBLEM_SIZE;j++){
      int node1=CFG[j];
      for_each_neighbor(node1,neibor){
	if(marked(neibor)){
	  assert(idx>PID[node1].newid);
	  set_adj_bit(idx,PID[node1].newid);
	  break;
	}
      }
      if(marked(node1)){
	for(int k=j+1;k<SUB_PROBLEM_SIZE;k++){
	  int node2=CFG[k];
	  if(marked(node2))
	    set_adj_bit(PID[node1].newid,PID[node2].newid);
	}
      }
    }
    clr_marked_status(node);
     for_each_neighbor(node,neibor){
       if(active(neibor) && involved(neibor))
	clr_marked_status(neibor);
    }
  }
  for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
     clr_involved_status(CFG[i]);
  }
}

static inline int profit_analysis(int isetno,int  marked_node){
  if(branched(marked_node)){
    for_each_vec_item(iSET[isetno],int,it){
      assert(marked_node!=*it);
      if(is_2_adj(marked_node,*it))
	return 0;
    }
    return 1;
  }
  int count=0;
  for_each_vec_item(iSET[isetno],int,it){
    if(marked(*it)){
      count++;
    }
  }
  if(count>1){
    return 1-count;
  }else if(count==0){
    for_each_vec_item(iSET[isetno],int,it){
      if(is_2_adj(marked_node,*it))
	return 0;
    }
    return 1;
  }
  assert(count==1);
  return 0;
}

static inline int insert_cur_node(int node){
  int inserted=0,lb=0;
  for(int j=0;j<MAXIS;j++){
    if(USED(iSET[j])==0){
      if(!inserted){
	set_isno(node,j);
	push_back(iSET[j],int,node);
	if(USED(iSET[j])>lb)
	  lb=USED(iSET[j]);
      }
      break;
    }
   
    if(iSET_Status[j]>0){
      inserted++;
      set_isno(node,j);
      push_back(iSET[j],int,node);
    }else if(iSET_Status[j]<0){
      int count=0;
      for_each_vec_item(iSET[j],int,it){
	if(marked(*it) && ++count>1){
	  *it=*(iSET[j]->addr+(--iSET[j]->used));
	  __end--,it--;
	}
      }
    }
    if(USED(iSET[j])>lb)
      lb=USED(iSET[j]);
  }
 return lb;
}



static void test_iset_consistance(){
  for(int j=0;j<MAXIS;j++){
    if(USED(iSET[j])==0)
      break;
    for_each_vec_item(iSET[j],int,it){
      int node1=*it;
      set_marked_status(node1);
      for_each_neighbor(node1,neibor){
	if(active(neibor) && involved(neibor))
	  set_marked_status(neibor);
      }
      for_each_vec_item(iSET[j],int,it2){
	int node2=*it2;
	if(node1<node2){
	  assert(is_2_adj(node1,node2)==0);
	}
      }
      clr_marked_status(node1);
      for_each_neighbor(node1,neibor){
	if(active(neibor) && involved(neibor))
	  clr_marked_status(neibor);
      }
    }
  }
}
static int repartition_vertices(int maxlb);
static int improve_partition(int maxlb,int target);

static int compute_initial_partition(){
  int maxlb=0;
  #ifdef CHECK
  for(int i=1;i<=NB_NODE;i++){
    assert(!involved(i));
    assert(!marked(i));
    assert(!branched(i));
    assert(!domed(i));
    assert(!fixed(i));
    assert(!removed(i));
  }
  #endif
  //clear matrix
  USED(ADJ_STK)=0;
  USED(FIX_STK)=0;
  //clear iset
  for(int j=0;j<=MAXIS;j++){
    USED(iSET[j])=0;
  }
  //testing each free vertex
  NEW_IDX=1;
  for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
    int cur_node=CFG[i];
    assert(!domed(cur_node));
 
    set_marked_status(cur_node);
    for_each_neighbor(cur_node,neibor){
      if(involved(neibor)){
       assert(active(neibor));
       assert(!domed(neibor)); 
       set_marked_status(neibor);
     }
    }
    
     int score=0;
    for(int j=0;j<MAXIS && USED(iSET[j]);j++){
      iSET_Status[j] = profit_analysis(j,cur_node);
      score += iSET_Status[j];
    }
    if(score>=0){
      set_involved_status(cur_node);
      set_newid(cur_node,NEW_IDX);
      push_back(FIX_STK,int,cur_node);
      //update_adj_matrix(cur_node,i,NEW_IDX);
      maxlb=insert_cur_node(cur_node);
      NEW_IDX++;
    }
    clr_marked_status(cur_node);
    for_each_neighbor(cur_node,neibor){
     if(involved(neibor))
	 clr_marked_status(neibor);
    }
  }
  
  if(maxlb<INIT_UPPER_BOUND){
    maxlb=repartition_vertices(maxlb);
  }
 
  if(maxlb<INIT_UPPER_BOUND){
    maxlb=improve_partition(maxlb,INIT_UPPER_BOUND);
  }
   
  if(maxlb<INIT_UPPER_BOUND){
    for_each_vec_item(FIX_STK,int,it){
      clr_involved_status(*it);
    }
  }
  return maxlb;
}

static int compute_lowerbound_with_2adj(){
  int maxlb=0;
  #ifdef CHECK
  for(int i=1;i<=NB_NODE;i++){
    assert(!involved(i));
    assert(!marked(i));
  }
  #endif
  //clear matrix
  USED(ADJ_STK)=0;
  USED(FIX_STK)=0;
  //clear iset
  for(int j=0;j<=MAXIS;j++){
    USED(iSET[j])=0;
  }
  //testing each free vertex
  NEW_IDX=1;
  for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
    int cur_node=CFG[i];
    assert(!domed(cur_node));    
    set_marked_status(cur_node);
    for_each_neighbor(cur_node,neibor){
      if(involved(neibor)){
	assert(active(neibor));
	assert(!domed(neibor));
	set_marked_status(neibor);
     }
    }
    
    int score=0;
    for(int j=0;j<MAXIS && USED(iSET[j]);j++){
      iSET_Status[j] = profit_analysis(j,cur_node);
      score += iSET_Status[j];
    }
    // printf("score=%d ",score);
    if(score>=0){
      set_involved_status(cur_node);
      // set_newid(cur_node,NEW_IDX);
      push_back(FIX_STK,int,cur_node);    
      maxlb=insert_cur_node(cur_node);
      NEW_IDX++;
    }
    // printf("\n");
    clr_marked_status(cur_node);
    for_each_neighbor(cur_node,neibor){
     if(involved(neibor))
	 clr_marked_status(neibor);
    }
  }
  #ifndef REP
  if(maxlb<BEST_LEVEL-CUR_LEVEL){
    for_each_vec_item(FIX_STK,int,it){
      clr_involved_status(*it);
    }
  }
  #endif
  return maxlb;
}
 


static int partition_free_vertices(){
  int lb=0,total=0;
  //clear iset
  for(int j=0;j<=MAXIS;j++){
    USED(iSET[j])=0;
  }
  for(int i=SUB_PROBLEM_SIZE-1,node=1;i>=CUR_UND_IDX;i--,node++){
    int inserted=0;
    for(int j=0;j<MAXIS;j++){
      if(USED(iSET[j])==0){
	if(!inserted){
	  set_isno(node,j);
	  push_back(iSET[j],int,node);
	  if(USED(iSET[j])>lb)
	    lb=USED(iSET[j]);
	}
	break;
      }
      int count=0;
      for_each_vec_item(iSET[j],int,it){
	if(is_2adj(node,*it)){
	  count=1;
	  break;
	}
      }
      if(count==0){
	inserted++;
	set_isno(node,j);
	push_back(iSET[j],int,node);
	if(USED(iSET[j])>lb)
	  lb=USED(iSET[j]);
      }
    }
    if(inserted==0){
      set_isno(node,MAXIS);
      push_back(iSET[MAXIS],int,node);
    }
    total++;
  }
  //  printf("  %d free vertice, lower bound = %d\n",total,lb);
  return lb;  
}

static inline  void clear_iset(int isetno,int insert_node){
  int involved=involved(insert_node);
  USED(TMP_STK)=0;
  for_each_vec_item(iSET[isetno],int,it){
    int node=*it;
    assert(!marked(node));
    for_each_neighbor(node,neibor){   
      if(active(neibor)&&involved(neibor)&&marked(neibor)){
	set_removed_status(neibor);
	push_back(TMP_STK,int,neibor);
      }
    }
  }
  for(int j=1;j<MAXIS;j++){
    if(USED(iSET[j])==0)
      break;
    if(j==isetno)
      continue;

    int cnt=0;
    for_each_vec_item(iSET[j],int,it){
      if(removed(*it)|| (marked(*it)&& !involved && ++cnt>1)){
	*it=*(iSET[j]->addr+(--iSET[j]->used));
	__end--,it--;
      }
    }
  }
   for_each_vec_item(TMP_STK,int,it){
  	clr_removed_status(*it);
	clr_involved_status(*it);
   }
}

static int improve_partition(int maxlb,int target){
  for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
    int node=CFG[i];

    set_marked_status(node);
    for_each_neighbor(node,neibor){   
      if(active(neibor))
	set_marked_status(neibor);
    }
    int newlb=0;
    for(int j=1;j<MAXIS;j++){
      if(USED(iSET[j])==0)
	break;

      if(USED(iSET[j])<maxlb)
	continue;
      
      int cnt=0;
      for_each_vec_item(iSET[j],int,it){
	if(marked(*it)){
	  cnt++;
	  break;
	}
      }
      
      if(!cnt){
	clear_iset(j,node);
	set_isno(node,j);
	push_back(iSET[j],int,node);
	set_involved_status(node);
	push_back(FIX_STK,int,node);
	if(USED(iSET[j])>newlb){
	  newlb=USED(iSET[j]);
	}
      }
    }
   
    clr_marked_status(node);
    for_each_neighbor(node,neibor){   
      if(active(neibor))
	clr_marked_status(neibor);
    }
    
    if(newlb>maxlb){
      maxlb=newlb;
      if (maxlb>=target){
	break;
      }
    }
  }
  return maxlb;
}

static int repartition_vertices(int maxlb){
  for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
    int node=CFG[i];
    
    if(!involved(node))
      continue;

    set_marked_status(node);
    for_each_neighbor(node,neibor){   
      if(active(neibor))
	set_marked_status(neibor);
    }

    for(int j=isno(node)+1;j<MAXIS;j++){
      if(USED(iSET[j])==0){
	break;
      }
      int count=0;
      for_each_vec_item(iSET[j],int,it){
	assert(node!=*it);
	if(is_2_adj(node,*it)){
	  count=1;
	  break;
	}
      }
      if(count==0){
	set_isno(node,j);
	push_back(iSET[j],int,node);
	if(USED(iSET[j])>maxlb){
	  maxlb=USED(iSET[j]);
	}
      }
    }
    clr_marked_status(node);
    for_each_neighbor(node,neibor){   
      if(active(neibor))
	clr_marked_status(neibor);
    }
  }
  return maxlb;
}



static int repartition_vertices2(int maxlb){
  for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
    int node=CFG[i];

    if(!involved(node))
      continue;

    int inserted=0;
    int new_idx=newid(node);
    for(int j=isno(node)+1;j<MAXIS;j++){
      if(USED(iSET[j])==0){
	break;
      }
      int count=0;
      for_each_vec_item(iSET[j],int,it){
	int new_idx2=newid(*it);
	assert(node!=*it);
	if(is_2adj(new_idx,new_idx2)){
	  count=1;
	  break;
	}
      }
      if(count==0){
	set_isno(node,j);
	push_back(iSET[j],int,node);
	if(USED(iSET[j])>maxlb){
	  maxlb=USED(iSET[j]);
	}
      }
    }
  }
  return maxlb;
}

static inline void update_2adj_matrix(){
  for(int i=CUR_UND_IDX,node1=NB_NODE-i;i<NB_NODE-1;i++,node1--){
    assert(newid(CFG[i])==node1);
    if(marked(node1)){
       for(int j=i+1,node2=NB_NODE-j;j<NB_NODE;j++,node2--){
	  assert(newid(CFG[j])==node2);
	 if(marked(node2)){
	  set_adj_bit(node1,node2);
	 }
       }
    }
  }
}


static int absorb_undomed_node(int node){

  int ret=FALSE;
  set_marked_status(node);
  for_each_neighbor(node,neibor){
    if(involved(neibor))set_marked_status(neibor);
  }
    
  int maxlb=0;
  for(int j=0;j<MAXIS && USED(iSET[j]);j++){
    iSET_Status[j] = profit_analysis(j,node);
    int size=USED(iSET[j])+iSET_Status[j];
    if(size>maxlb)maxlb=size;
  }
  if(maxlb>=BEST_LEVEL-CUR_LEVEL){
    set_involved_status(node);
    set_newid(node,NEW_IDX);
    push_back(FIX_STK,int,node);
    //update_adj_matrix(node,i,NEW_IDX);
    int lb=insert_cur_node(node);
    assert(maxlb==lb);
    NEW_IDX++;
    ret=TRUE;
  }
  clr_marked_status(node);
  for_each_neighbor(node,neibor){
    if(involved(neibor))clr_marked_status(neibor);
  }
  return ret;
}

static int absorb_domed_node(int node){
  int maxLB=0;
  // printf("\n  $$testing %d with neibors \n",node);
  assert(domed(node));
  
  for_each_neighbor(node,neibor){
    if(active(neibor) && involved(neibor)){
      set_marked_status(neibor);
    }
  }
  // printf("\n");
  for(int j=0;j<MAXIS;j++){
    if(USED(iSET[j])==0)
      break;
    // printf("iSET %d Size %d \n",j,USED(iSET[j]));
    int count=0,size=USED(iSET[j]);
    for_each_vec_item(iSET[j],int,it){
      //printf("(%d) ",CFG[NB_NODE-*it]);
      if(marked(*it)){
        //printf("<%d>",CFG[NB_NODE-*it]);
	count++;
      }
    }
    iSET_Status[j]=count;
    // printf("\n");   
    if(count>1)
      size=size-count+1;
    if(size>maxLB)
      maxLB=size;
   
    //printf("iSET %d Size %d \n",j,size);
  }

  // printf("  LB=%d BEST_LEVEL = %d CUR_LEVEL =%d\n",maxLB,BEST_LEVEL,CUR_LEVEL);
  if(maxLB<BEST_LEVEL-CUR_LEVEL){
    for_each_neighbor(node,neibor){
      if(active(neibor) && involved(neibor)){
	clr_marked_status(neibor);
      }
    }
    //  printf("  **%d is active\n ",node);
    return FALSE;
  }
  maxLB=0;
  for(int j=0;j<MAXIS;j++){
    if(iSET_Status[j]>1){
      int count=0;
      for_each_vec_item(iSET[j],int,it){
	if(marked(*it) && ++count>1){
	  *it=*(iSET[j]->addr+(--iSET[j]->used));
	  __end--,it--;
	}
      }
    }
    if(USED(iSET[j])>maxLB)
      maxLB=USED(iSET[j]);
  }
  assert(maxLB>=BEST_LEVEL-CUR_LEVEL);

 
  // #ifdef CHECK
  set_involved_status(node);
  push_back(FIX_STK,int,node);
  //#endif
  /*
  for(int i=CUR_UND_IDX;i<SUB_PROBLEM_SIZE-1;i++){
    // assert(newid(CFG[i])==node1);
    int node1=CFG[i];
    if(marked(node1)){
      for(int j=i+1;j<SUB_PROBLEM_SIZE;j++){
	int node2=CFG[j];
	if(marked(node2)){
	  set_adj_bit(newid(node1),newid(node2));
	}
      }
    }
  }
  */
  // printf("clear mark ");
  for_each_neighbor(node,neibor){
    if(active(neibor) && involved(neibor)){
      // printf("%d ",neibor);
       clr_marked_status(neibor);
    }
  }
  // printf("  >>%d is absorbed @%d\n",node,CUR_LEVEL);
  return TRUE;
}

static void reduce_dominated_vertice(){  
  for(int i=CUR_LEVEL;i<CUR_UND_IDX;i++){
    int node=CFG[i];
    assert(domed(node));
    if(!branched(node)
       && absorb_domed_node(node)==FALSE)
      push_back(BRA_STK,int,node);
  }
  push_back(BRA_STK,int,NONE);
}

#define BNODE(i) ITEM(BRA_STK,i)

static int dominated_number(int bnode){
    int count=0;
    assert(!branched(bnode));
    if(!domed(bnode))
      count=1;
    for_each_neighbor(bnode,neibor){
      if(active(neibor) && !domed(neibor))
	count++;
    }
    return count;
}

int max_dominated_number(){
  int max_count=0;
  for(int i=CUR_BRA_IDX,bnode=BNODE(i);bnode!=NONE;bnode=BNODE(++i)){
    int count= dominated_number(bnode);
    if(count>max_count)
      max_count=count;
  }
  return max_count;
}

int select_branching_node(){
  int max_count=0,max_idx=-1;
  for(int i=CUR_BRA_IDX,bnode=BNODE(i);bnode!=NONE;bnode=BNODE(++i)){
    int count=0;
    assert(!branched(bnode));
    if(!domed(bnode))
      count=1;
    for_each_neighbor(bnode,neibor){
      if(active(neibor) && !domed(neibor))
	count++;
    }
    if(count>max_count){
      max_count=count;
      max_idx=i;
    }
  }

  //assert(max_idx!=-1);
  
  if(max_idx==-1 && BEST_LEVEL==SUB_PROBLEM_SIZE && CUR_UND_IDX<SUB_PROBLEM_SIZE){
     printf("\nthe graph is *unconnected*!\n");
     exit(0);
  }
    
  if(BEST_LEVEL==SUB_PROBLEM_SIZE && CUR_UND_IDX<SUB_PROBLEM_SIZE){
    assert(max_idx>=0);
  }
  // printf("choosing %d\n",BNODE(max_idx));
  if(max_idx>=0 && CUR_BRA_IDX!=max_idx){
    int temp=BNODE(CUR_BRA_IDX);
    BNODE(CUR_BRA_IDX)=BNODE(max_idx);
    BNODE(max_idx)=temp;
  }
  return BNODE(CUR_BRA_IDX);
}

static void push_fixed_vertices(){
  int fnode,prenode=0;
  CUR_LEVEL=-1;
  CUR_BRA_IDX=0;
  for(int i=0;i<USED(FIX_STK);i++){
    fnode=ITEM(FIX_STK,i);
    assert(!branched(fnode) && fixed(fnode));
    CUR_LEVEL++;
    CUR_BRA_IDX=USED(BRA_STK);
    push_back(BRA_STK,int,NONE);
    push_back(BRA_STK,int,fnode);
    // set_und_idx(fnode,CUR_UND_IDX);
    CUR_LEVEL_UND_IDX=CUR_UND_IDX;
    
    if(prenode){
      if(!domed(prenode)){
	swap_cfg(CFG[CUR_UND_IDX],prenode);
	CUR_UND_IDX++;
      }

      CUR_LEVEL--;
      CUR_BRA_IDX++;
      CUR_LEVEL++;

      set_domed_status(prenode);
      set_branched_status(prenode); 
      for_each_neighbor(prenode,neibor){
	if(active(neibor) && !domed(neibor)){
	  int first=CFG[CUR_UND_IDX];
	  if(first!=neibor)
	    swap_cfg(first,neibor);
	  set_domed_status(neibor);
	  CUR_UND_IDX++;
	}
      }
    }
    prenode=fnode;
  }
  push_back(BRA_STK,int,NONE);
}
#define MAX_SCORE 10000
static inline void update_score(int bnode,int score){
  if(Node_Degree[bnode]+score>=MAX_SCORE){
    for(int i=1;i<=NB_NODE;i++){
      Node_Degree[bnode]=Node_Degree[bnode]/10;
    }
  }
  Node_Degree[bnode]+=score;
}


static inline void save_and_backtrack(int bnode){
  if(CUR_LEVEL<BEST_LEVEL){
    BEST_LEVEL=CUR_LEVEL;
    #ifdef CHECK
      check_and_save_solution();
    #else
      update_best_solution();
    #endif
  }
  rollback_branch_node(bnode);
  while(ITEM(BRA_STK,++CUR_BRA_IDX)!=NONE);    
  backtrack();
}

static inline void sorting_branch_vertices(int startidx){
 int length=0;
  for(int i=startidx,bnode=BNODE(i);bnode!=NONE;bnode=BNODE(++i)){
    int count=0;
    assert(!branched(bnode));
    if(!domed(bnode))
      count=1;
    for_each_neighbor(bnode,neibor){
      if(active(neibor) && !domed(neibor))
	count++;
    }
    Node_Degree[bnode]=count;
    length++;
  }
  qsort(BRA_STK->addr+startidx,length,sizeof(int),cmp_branching_vertex_score);
}

void search_domset(){
  int bnode;
  #ifdef INFO
  print_config(1);
  print_branch_stack();
  #endif
  while(CUR_LEVEL>=0){
    #if CHECK
    check_consistance();
    #endif    
    if((bnode=CUR_BRA_NODE)!=NONE){
      rollback_branch_node(bnode);     
    }    
  next:
    CUR_BRA_IDX++;
    #ifdef INFO
    print_config(1);
    print_branch_stack(); 
    #endif 
    //backtrack
    if(CUR_BRA_NODE==NONE){
    #ifdef INFO
      printf("  <-- backtracking to level %d ...\n",CUR_LEVEL-1);
    #endif 
      backtrack();
      continue;
    }

    #ifdef MGS
    bnode=select_branching_node();
    #else
    bnode=CUR_BRA_NODE;
    #endif
    NB_TREE++;
    assert(!branched(bnode));
    #ifdef INFO
    printf("  --> branching on %d...\n",bnode);
    #endif 
    set_branched_status(bnode);

    int domc=0;
    if(!domed(bnode)){
      swap_cfg(CFG[CUR_UND_IDX],bnode);
      set_domed_status(bnode);
      CUR_UND_IDX++;
      domc++;
    }
      
    for_each_neighbor(bnode,neibor){
      if(active(neibor) && !domed(neibor)){
	domc++;
	assert(!fixed(neibor));
	int first=CFG[CUR_UND_IDX];
	if(first!=neibor)
	  swap_cfg(first,neibor);
	set_domed_status(neibor);
	CUR_UND_IDX++;
      }
    }
    
    if(!domc){
      assert(fixed(bnode));
    }
    
    assert(CUR_UND_IDX<=SUB_PROBLEM_SIZE);
   
    if(CUR_UND_IDX>=SUB_PROBLEM_SIZE){
      save_and_backtrack(bnode);
      continue;
    }
    #ifndef NOLB    
      int lb= compute_lowerbound_with_2adj();
      #ifdef REP
        if(lb<BEST_LEVEL-CUR_LEVEL){
          lb=repartition_vertices(lb);
        }
        if(lb<BEST_LEVEL-CUR_LEVEL){
          lb=improve_partition(lb,BEST_LEVEL-CUR_LEVEL);
        }
        if(lb<BEST_LEVEL-CUR_LEVEL){
          for_each_vec_item(FIX_STK,int,it){
	    clr_involved_status(*it);
          }
        }
      #endif
      #ifdef CHECK
        test_iset_consistance();
      #endif
    #else
      for(int i=SUB_PROBLEM_SIZE-1;i>=CUR_UND_IDX;i--){
	 clr_involved_status(CFG[i]);
      }
    #endif
   
   
    #ifdef INFO
      // printf("         lb=%d expected >= %d\n",lb,BEST_LEVEL-CUR_LEVEL);
    #endif
    #ifdef INFO
       print_each_iset();
    #endif
    #ifndef NOLB
    int  succ = (lb>=BEST_LEVEL-CUR_LEVEL)? 1:0;
    #else
    int  succ = 0;
    #endif
    //go to next level
    int start_idx=USED(BRA_STK)-1,nb_branches=0, nb_total=0;

    int *ptr=CFG;
    for(int node=*ptr;node!=NONE;node=*(++ptr)){      
      if(branched(node))
	continue;
      if(succ && involved(node))
	continue;
      
      int count=0;
      if(domed(node)){
	for_each_neighbor(node,neibor){
	  if(active(neibor) && !domed(neibor)){
	    count++;
	    break;
	  }
	}
	if(!count){
	  if(++start_idx==USED(BRA_STK)){
	    push_back(BRA_STK,int,node);
	    assert(BNODE(start_idx)==node);
	  }else{
	    int temp=BNODE(start_idx);
	    BNODE(start_idx)=node;
	    push_back(BRA_STK,int,temp);
	  }
	  set_branched_status(node);
	  continue;
	}
      }
      nb_total++;
      
      if(!succ){
	push_back(BRA_STK,int,node);
	nb_branches++;
      }else if(domed(node)){
        if(absorb_domed_node(node)==FALSE){
	  push_back(BRA_STK,int,node);
	  nb_branches++;
	}
      }else if(absorb_undomed_node(node)==FALSE){
	assert(!domed(node)&&!involved(node));
	push_back(BRA_STK,int,node);
	nb_branches++;
      }
    }
    #ifdef REP
     for_each_vec_item(FIX_STK,int,it){
	clr_involved_status(*it);
      }
    #else
      if(succ){
      for_each_vec_item(FIX_STK,int,it){
	clr_involved_status(*it);
      }
    }
    #endif
    
#if CHECK
    for(int i=0;i<SUB_PROBLEM_SIZE;i++){
	assert(!involved(CFG[i]));
      }
#endif
     total_branches+=nb_total;
     pruned_branches+=nb_branches;
   
    if(nb_branches && CUR_LEVEL<BEST_LEVEL){ 
      CUR_LEVEL++;
      CUR_BRA_IDX=start_idx;
      CUR_LEVEL_UND_IDX=CUR_UND_IDX;
      push_back(BRA_STK,int,NONE);
      #ifndef MGS
      sorting_branch_vertices(start_idx+1);
      #endif
    }else{
      for(int node=BNODE(start_idx);node!=NONE;node=BNODE(--start_idx)){
	clr_branched_status(node);
      }
      USED(BRA_STK)=start_idx+1;
    }
  }
}


static MaxHeap HeapForIni;
int fast_search_initial_solution(){
  int *Val= malloc((NB_NODE+1)*sizeof (int));
  memset(Val,0,(NB_NODE+1)*sizeof (int));
  USED(VEC_SOLUTION)=0;
  Node temp;
  MaxHeap *Que=&HeapForIni;
  initHeap(&HeapForIni,SUB_PROBLEM_SIZE,node_cmp_for_MaxHeap);
  int *ptr=CFG,*end=CFG+SUB_PROBLEM_SIZE,NB_Domed=0;
  for(int node=*ptr;ptr<end;node=*(++ptr)){
    assert(active(node));
    if(fixed(node)){
      push_back(VEC_SOLUTION,int,node);
      set_branched_status(node);
      for_each_neighbor(node,neibor){
	if(!active(neibor) || node==neibor)continue;
	if(!domed(neibor)){
	  NB_Domed++;
	  set_domed_status(neibor);
	}
      }
      if(!domed(node)){
	NB_Domed++;
	set_domed_status(node);
      }
    }
  }
  ptr=CFG,end=CFG+SUB_PROBLEM_SIZE;
  for(int node=*ptr;ptr<end;node=*(++ptr)){
    if(fixed(node))continue;
    if(!domed(node))Val[node]=1;
    else Val[node]=0;
    for_each_neighbor(node,neibor){
      if(neibor==node)continue;
      if(active(neibor) && !domed(neibor)){
	Val[node]++;
      }
    }
    temp.ID=node;
    temp.V=Val[node];
    insertHeap(Que,temp);
  }
  int bnode;
  while(NB_Domed<SUB_PROBLEM_SIZE){
    temp=removeTop(Que);
    bnode=temp.ID;
    if (temp.V!=Val[bnode]){
      temp.V=Val[bnode];
      insertHeap(Que,temp);
      continue;
    }
    assert(bnode);
    if(branched(bnode))continue;
    assert(!fixed(bnode));
    assert(active(bnode));
    assert(Val[bnode]>0 && Val[bnode]<=Node_Degree[bnode]+1);
    push_back(VEC_SOLUTION,int,bnode);
    set_branched_status(bnode);
    for_each_neighbor(bnode,neibor){
      if(!active(neibor) || bnode==neibor)continue;
      if(!domed(neibor)){
	NB_Domed++;
	set_domed_status(neibor);
	for_each_neighbor(neibor,nei)if(active(nei) && !fixed(nei) && nei!=neibor)Val[nei]--;
	Val[neibor]--;
      }
    }
    if(!domed(bnode)){
      NB_Domed++;
      set_domed_status(bnode);
      for_each_neighbor(bnode,nei)if(active(nei) && !fixed(nei)&& !branched(nei))Val[nei]--;
    }
    Val[bnode]=0;
  }
  int *pptr=CFG,*eend=CFG+SUB_PROBLEM_SIZE;
  for(int node=*pptr;pptr<eend;node=*(++pptr)) {
    clr_domed_status(node);
    clr_branched_status(node);
  }
  free(Val);free(Que->array);
  return USED(VEC_SOLUTION);

}

static int search_initial_solution(){
  USED(VEC_SOLUTION)=0;
  while(1){
    int *ptr=CFG,*end=CFG+SUB_PROBLEM_SIZE;
    int max=0,bnode=0;
    for(int node=*ptr;ptr<end;node=*(++ptr)){
      int  count=0;
      if(branched(node))
	continue;

      if(fixed(node)){
	max=1;
	bnode=node;
	break;
      }
	
      if(!domed(node))
	count=1;
      
      for_each_neighbor(node,neibor){
	if(active(neibor) && !domed(neibor))
	  count++;
      }
      if(count>max){
	max=count;
	bnode=node;
      }
    }
    if(max>0){
      assert(bnode);
      push_back(VEC_SOLUTION,int,bnode);
      set_domed_status(bnode);
      set_branched_status(bnode);
      for_each_neighbor(bnode,neibor){
	set_domed_status(neibor);
      }          
    }else{
      int *ptr=CFG,*end=CFG+SUB_PROBLEM_SIZE;
      for(int node=*ptr;ptr<end;node=*(++ptr)){
	clr_domed_status(node);
	clr_branched_status(node);
      }
      
      return USED(VEC_SOLUTION);
    }
  }
  return USED(VEC_SOLUTION);
}

void solve_subproblems(){

  int i=0,pro_no=1;
  int *ptr=VEC_SUBGRAPHS->addr-1;
  int *end=VEC_SUBGRAPHS->addr+USED(VEC_SUBGRAPHS)-1;

  do{
    int *start=ptr;
    int j=0,first_node=0,fixed_count=0;
    USED(FIX_STK)=0;
    for(int node=*(++ptr);node>0;node=*(++ptr)){
      CFG[j]=node;
      LOC[node]=j++;
      set_active(node);
      clr_domed_status(node);
      clr_branched_status(node);
      if(fixed(node)){
	fixed_count++;
	push_back(FIX_STK,int,node);
      }
    }
    CFG[j]=NONE;
    SUB_PROBLEM_SIZE=j;
   
    if(INIT_UPPER_BOUND==0){
      #ifndef NOFAST
       INIT_UPPER_BOUND=fast_search_initial_solution();
      #else
       INIT_UPPER_BOUND=search_initial_solution();
      #endif
    }
    BEST_LEVEL = INIT_UPPER_BOUND-1;

    BRAIDX=(int *)malloc((INIT_UPPER_BOUND+1)*sizeof(int));
    UNDIDX=(int *)malloc((INIT_UPPER_BOUND+1)*sizeof(int));

    USED(BRA_STK)=0;
   
    if(USED(FIX_STK)){
      push_fixed_vertices();
    }else{
      CUR_LEVEL=0;
      CUR_BRA_IDX=0;
      CUR_UND_IDX=0;
      CUR_LEVEL_UND_IDX=0;
      push_back(BRA_STK,int,NONE);
      #ifndef NOLB
      compute_initial_partition();
      #endif
      for(int i=0;i<SUB_PROBLEM_SIZE;i++){
	if(!involved(CFG[i]))
	  push_back(BRA_STK,int,CFG[i]);
	else
	  clr_involved_status(CFG[i]);
      }
      push_back(BRA_STK,int,NONE);
      sorting_branch_vertices(1);
    }
    
    search_domset();
  
    for(int node=*(++start);node>0;node=*(++start)){
      clr_active(node);
      clr_domed_status(node);
      clr_branched_status(node);
    }
   
  }while(ptr<end);
}

// Define global variables
extern unsigned long long total_branches = 0;
extern unsigned long long pruned_branches = 0;

// Move all function implementations from main-modified.c here
void check_final_solution(){
  for_each_vertex(node){
    clr_marked_status(node);
    clr_included_status(node);
  }
  for_each_vec_item(VEC_SOLUTION,int,it){
     int node=*it;
     assert(node>=1 && node<=NB_NODE);
     set_marked_status(node);
     assert(!included(node));
     set_included_status(node);
    for_each_neighbor(node,neibor){
        set_marked_status(neibor);
    }
  }
 
  for_each_vertex(node){
   
    if(deleted(node))continue;
   
    assert(marked(node));
    clr_marked_status(node);
  }

  for_each_vertex(node){
     if(deleted(node)){
      int flag=0;
      for_each_neighbor(node,neibor){
        if(included(neibor)){
	  flag=1;
	  break;
	}
      }
      assert(flag);
     }
   }

}


void update_best_solution(){
  USED(VEC_SOLUTION)=0;
  for(int i=0;i<=CUR_LEVEL;i++){
    int node=branch_node_at_level(i);
    push_back(VEC_SOLUTION,int,node);
  }
}

void check_and_save_solution(){
  USED(VEC_SOLUTION)=0;
  for_each_vertex(node){
    clr_marked_status(node);
    clr_included_status(node);
  }
  for(int i=0;i<=CUR_LEVEL;i++){
    int node=branch_node_at_level(i);
    assert(!included(node));
    set_included_status(node);
    push_back(VEC_SOLUTION,int,node);
    set_marked_status(node);
    for_each_neighbor(node,neibor){
      if(active(neibor))
        set_marked_status(neibor);
    }
  }
  for_each_vertex(node){
    if(active(node)){
       assert(marked(node));
       clr_marked_status(node);
    }
  }
}

void check_consistance(){
  
  for(int i=0;i<SUB_PROBLEM_SIZE;i++){
    assert(CFG[i]>=1 && CFG[i]<=NB_NODE);
    assert(LOC[CFG[i]]==i);
    assert(!involved(CFG[i]));
  }
  
  for(int i=0;i<CUR_UND_IDX;i++){
    assert(CFG[i]>=1 && CFG[i]<=NB_NODE);
    assert(domed(CFG[i]));
  }

  for(int i=CUR_UND_IDX;i<SUB_PROBLEM_SIZE;i++){
    assert(CFG[i]>=1 && CFG[i]<=NB_NODE);
    assert(!domed(CFG[i]));
  }
  int level=-1;
  for(size_t idx=0;idx<USED(BRA_STK);idx++){
    if(ITEM(BRA_STK,idx)==NONE){
      level++;
    }else if(idx<=BRAIDX[level])
      assert(branched(ITEM(BRA_STK,idx)));
  }
  
  for(int i=0;i<CUR_LEVEL;i++){
    int node=branch_node_at_level(i);
     assert(node>=1 && node<=NB_NODE);
     assert(domed(node));
    for_each_neighbor(node,neibor){
      if(!active(neibor))
	continue;
      assert(domed(neibor));
      assert(active(neibor));
      assert(LOC[neibor]<CUR_UND_IDX);
    }
  }
  // printf("pass consistance checking at level %d ...\n",CUR_LEVEL);
  fflush(stdout);
}


void cleanup(){

  // Free all allocated memory
  if (CFG != NULL) {
    free(CFG);
    CFG = NULL;
  }
  if (LOC != NULL) {
    free(LOC);
    LOC = NULL;
  }
  if (STATUS != NULL) {
    free(STATUS);
    STATUS = NULL;
  }
  if (PID != NULL) {
    free(PID);
    PID = NULL;
  }
  free_stack(BRA_STK);
  free_stack(FIX_STK);
  free_stack(TMP_STK);
  free_stack(ADJ_STK);
  free_stack(VEC_SOLUTION);
  free_stack(VEC_SUBGRAPHS);

  for (int i = 0; i <= MAXIS; i++) {
    free_stack(iSET[i]);
  }

  if (ADJIDX != NULL) {
    free(ADJIDX);
    ADJIDX = NULL;
  }



  free_block();
  // Reset all global variables
  BLOCK_COUNT = 0;
  NB_NODE = 0;
  NB_EDGE = 0;
  Max_Degree = 0;
  Max_Degree_Node = 0;
  SUB_PROBLEM_SIZE = 0;
  FORMAT = 1;
  NB_NODE_O = 0;
  NB_EDGE_O = 0;
  READ_TIME = 0;
  INIT_TIME = 0;
  SEARCH_TIME = 0;
  D0 = 0;
  D1 = 0;
  D2 = 0;
  Dt = 0;
  INIT_BRANCHING_NODE = 0;
  INIT_UPPER_BOUND = 0;
  NB_TREE = 0;
  BEST_LEVEL = 0;
  CUR_LEVEL = 0;
  CUR_UND_IDX = 0;
  BEST_SOL_TIME = 0;
  TIME_OUT = 0;
  CUT_OFF = 0;
  BEST_SOL_TIME = 0;
  instance[0] = '\0';
}


void handler(int sig) {
  cleanup();
  exit(sig);
}

#ifdef _WIN32
static BOOL WINAPI win32_handler(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        handler(SIGINT);
        return TRUE;
    }
    return FALSE;
}

static void setup_signal_handler(SignalHandlerFn handler_fn) {
    SetConsoleCtrlHandler(win32_handler, TRUE);
}
#else
static void setup_signal_handler(void (*handler_fn)(int)) {
    signal(SIGINT, handler_fn);
}
#endif


struct Result* emos_main(unsigned int* edges, int n, int nb_edge) {

    // Set the signal handler
  setup_signal_handler(handler);

  _read_graph_from_edge_list(edges, n, nb_edge);
  NB_NODE_O = NB_NODE;
  NB_EDGE_O = NB_EDGE;
  D0 = ((float) NB_EDGE * 2 / NB_NODE / (NB_NODE - 1));
  READ_TIME = get_utime();

  // initialize other EMOS data structures
  initialize();

  double start_time = get_utime();
  #ifndef NOR
  reduce_graph();
  #endif
  partition_oneproblem();
  solve_subproblems();
  check_final_solution();
  double exec_time = get_utime() - start_time;

  // Get the results
  int* dominating_set = (int*)malloc(USED(VEC_SOLUTION) * sizeof(int));
  for (size_t i= 0; i<USED(VEC_SOLUTION); i++) {
    dominating_set[i] = ITEM(VEC_SOLUTION, i);
  }

  struct Result* result = (struct Result*)malloc(sizeof(struct Result));
  result->dominating_set = dominating_set;
  result->set_size = USED(VEC_SOLUTION);
  result->exec_time = exec_time;
  
  return result;
}


int* get_dominating_set(struct Result* result) {
    return result->dominating_set;
}

int get_set_size(struct Result* result) {
    return result->set_size;
}

double get_exec_time(struct Result* result) {
    return result->exec_time;
}

void free_results(struct Result* result) {
    if (result) {
        free(result->dominating_set);
        free(result);
    }
}
	
