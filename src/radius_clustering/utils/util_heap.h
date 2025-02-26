#include <stdlib.h>
#include <assert.h>

#ifndef V6R_UTIL_HEAP_H
#define V6R_UTIL_HEAP_H

typedef struct MaxNodeHeap MaxHeap;
typedef struct node Node;

static const int INF=0x3f3f3f3f;
struct node {
    int ID,V;
};

struct MaxNodeHeap{
    unsigned size;
    unsigned capacity;
    Node *array;
    int (*comparator)(const Node * ,const Node *);
};


#define size_of_heap(H)  (H->size)
#define empty_heap(H)  (H->size==0)

static inline unsigned int left  (int i) { return i*2+1; }
static inline unsigned int right (int i) { return (i+1)*2; }
static inline unsigned int parent(int i) { return (i-1) >> 1; }

static inline void shiftUp(MaxHeap *heap,int i)
{
    Node x  = heap->array[i];
    int p  = parent(i);
    while (i != 0 && heap->comparator(&x, &(heap->array[p]))){
        heap->array[i]= heap->array[p];
        i = p;
        p = parent(p);
    }
    heap->array[i] = x;
}

static inline void shiftDown(MaxHeap *heap,int i)
{
    assert(heap->comparator);

    Node x = heap->array[i];
    while (left(i) < heap->size){
        int child = (right(i) < heap->size && heap->comparator(&(heap->array[right(i)]),&(heap->array[left(i)]))) ? right(i) : left(i);
        if (!heap->comparator(&(heap->array[child]), &x)) break;
        heap->array[i] = heap->array[child];
        i = child;
    }
    heap->array[i] = x;
}

static inline void insertHeap(MaxHeap *heap,Node x){
    assert(heap->size<=heap->capacity);
    if(heap->size==heap->capacity) {
        int NewSize=2*(heap->capacity);
        heap->array=(Node *)(realloc(heap->array,(NewSize+1)*sizeof (Node)));
        assert((heap->array)!=NULL);
        heap->capacity=NewSize;
    }
    heap->array[heap->size]=x;
    heap->size++;
    if(heap->size>1)shiftUp(heap,heap->size-1);
}

static inline int node_cmp_for_MaxHeap(const Node *A,const Node *B) {
    if(A->V==B->V)return A->ID<B->ID;
    return A->V > B->V;
}

static inline void initHeap(MaxHeap *heap,int capacity,int (*cmp)(const Node *, const Node*)){
    heap->array=(struct node *)calloc(capacity+1,sizeof(Node));
    heap->capacity=capacity;
    heap->size=0;
    heap->comparator=cmp;
}

static inline Node removeTop(MaxHeap *heap)
{

    Node x = heap->array[0];
    heap->array[0]  = heap->array[heap->size-1];
    heap->size--;
    if (heap->size > 1) shiftDown(heap, 0);
    return x;
}

static inline void clearHeap(MaxHeap *heap){
    heap->size=0;
}
#endif //V6R_UTIL_HEAP_H
