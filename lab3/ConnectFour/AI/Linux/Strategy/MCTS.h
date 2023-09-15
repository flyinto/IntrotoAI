#include "Judge.h"
#include "Point.h"
#include <cmath>
#include <ctime>

class Node
{
private:
    int **board;    //当前棋盘状态
    int M, N;   //行数M，列数N
    int noX, noY;   //禁止落子点坐标
    int lastX, lastY;   //对手最后落子点坐标
    int *top;   //当前顶端状态，top[i]表示第i列顶端行标
    int player;     //当前执棋方，1为玩家(对手)，2为AI(己方)
    int win;    //胜利次数(针对当前执棋方而言)
    int visit;  //访问总次数
    Node* parent;   //父节点
    Node** child;   //子节点列表
    int expand_num;     //可扩展子节点数(可决策数)
    int* expand_column;  //可扩展列表(可以下在哪几列)
    void copy_board(int **newboard) {
        newboard = new int*[M];
        for(int i = 0; i < M; i++) {
            newboard[M] = new int[N];
            for(int j = 0; j < N; j++) newboard[i][j] = board[i][j];
        }
    }   //复制当前棋盘
    void copy_top(int *newtop) {
        newtop = new int[N];
        for(int i = 0; i < N; i++) newtop[i] = top[i];
    }   //复制top数组

friend class UCT;

public:
    Node(int **Board, int r, int c, int nox, int noy, int lastx, int lasty, int *Top, int Player, Node* p):
    board(Board), M(r), N(c), noX(nox), noY(noy), lastX(lastx), lastY(lasty), top(Top), player(Player), parent(p) {
        win = visit = expand_num = 0;
        expand_column = new int[N];
        for(int i = 0; i < N; i++) {
            if(top[i] > 0) expand_column[expand_num++] = i;
        }
        child = new Node*[expand_num];
        for(int i = 0; i < expand_num; i++) child[i] = nullptr;
    }   //构造函数

    ~Node() {
        for(int i = 0; i < M; i++) {
            delete[] board[i];
        }
        delete[] board;
        delete[] top;
        for(int i = 0; i < expand_num; i++) {
            if(child[i]) delete child[i];
        }
        delete[] child;
        delete[] expand_column;
        //std::cerr << "Successfully destruct a node" << std::endl;
    }   //析构函数

    bool is_expandable() {
        if(!expand_num) return false;
        for(int i = 0; i < expand_num; i++) {
            if(child[i] == nullptr) return true; 
        }
        return false;
    }   //判断是否仍有待扩展的子节点

    bool is_terminal() {
        if(lastX == -1 && lastY == -1) return false;
        return (player == 1 && machineWin(lastX, lastY, M, N, board)) 
        || (player == 2 && userWin(lastX, lastY, M, N, board))
        || (isTie(N, top));
    }   //判断是否为终局

    Node* best_child(double c) {
        Node* best = nullptr;
        double max_uc = -1e18;
        for(int i = 0; i < expand_num; i++) {
            if(child[i] == nullptr) continue;
            double rate = -1.0 * child[i]->win / child[i]->visit;
            double modify = c * sqrt(2 * log(visit) / log(child[i]->visit));
            if(rate + modify > max_uc) {
                max_uc = rate + modify;
                best = child[i];
            }
        }
        return best;
    }   //返回信心上限最大的孩子

    Node* expand() {
        int *decision = new int[expand_num];
        int rest = 0;
        for(int i = 0; i < expand_num; i++)
            if(child[i] == nullptr) decision[rest++] = i;
        //std::cerr << "Successfully get expandable children" << std::endl;
        int **newboard = new int*[M];
        for(int i = 0; i < M; i++) {
            newboard[i] = new int[N];
            for(int j = 0; j < N; j++) newboard[i][j] = board[i][j];
        }
        int *newtop = new int[N];
        for(int i = 0; i < N; i++) newtop[i] = top[i];
        //std::cerr << "Successfully copy the states" << std::endl;
        int index = decision[rand() % rest], y = expand_column[index], x = top[y] - 1;
        //std::cerr << "Successfully choose child" << std::endl;
        newboard[x][y] = player, newtop[y]--;
        child[index] = new Node(newboard, M, N, noX, noY, x, y, newtop, 3 - player, this);
        //std::cerr << "Successfully new child" << std::endl;
        delete[] decision;
        //std::cerr << "Successfully expand node" << std::endl;
        return child[index];
    }   //节点扩展
};

class UCT
{
private:
    Node* root;
    int M, N;
    int noX, noY;
    double coef;
    int start_time;

public:
    UCT(int r, int c, int nox, int noy, double co) : M(r), N(c), noX(nox), noY(noy), coef(co), start_time(clock()) {}

    ~UCT() {delete this->root;}

    Node* TreePolicy(Node* v) {
        while(!v->is_terminal()) {
            if(v->is_expandable()) {
                //std::cerr << "Try to expand node" << std::endl;
                return v->expand();
            }
            else v = v->best_child(coef);
            //std::cerr << "Current node has been fully expanded, goes to best child" << std::endl;
        }
        return v;
    }   //找到目标节点并扩展，如果出现终态直接回溯更新

    void RollPlay(int **board, int *top, int player, int &x, int &y) {
        y = rand() % N; int left = (y < (N >> 1));
        while(top[y] <= 0 && y >= 0 && y < N) left ? y++ : y--;
        if(y < 0 || y >= N) return ;
        x = --top[y];
        board[x][y] = player;
        if(x - 1 == noX && y == noY) top[y]--;
    }   //随机模拟

    int GetWin(int **board, int *top, int player, int x, int y) {
        if((player == 1 && userWin(x, y, M, N, board)) || (player == 2 && machineWin(x, y, M, N, board)))
            return 1;
        else if((player == 2 && userWin(x, y, M, N, board)) || (player == 1 && machineWin(x, y, M, N, board)))
            return -1;
        else if(isTie(N, top))
            return 0;
        else return 114514;
    }   //判断Player在当前局面下是否已经获胜

    int DefaultPolicy(Node* node) {
        int **Board = new int*[M];
        for(int i = 0; i < M; i++) {
            Board[i] = new int[N];
            for(int j = 0; j < N; j++) Board[i][j] = node->board[i][j];
        }
        int *Top = new int[N];
        for(int i = 0; i < N; i++) Top[i] = node->top[i];
        int x = node->lastX, y = node->lastY, player = node->player;
        while(GetWin(Board, Top, node->player, x, y) == 114514) {
            RollPlay(Board, Top, player, x, y);
            player = 3 - player;
        }
        //std::cerr << "Random rollplay has finished" << std::endl;
        int profit = GetWin(Board, Top, node->player, x, y);
        for(int i = 0; i < M; i++) delete[] Board[i];
        delete[] Board;
        delete[] Top;
        return profit;
    }

    void BackUp(Node *v, int delta) {
        while(v != nullptr) {
            v->visit++;
            v->win += delta;
            delta = -delta;
            v = v->parent;
        }
        //std::cerr << "Profits have been updated" << std::endl;
    }

    int UCTSearch(int **board, int *top) {
        root = new Node(board, M, N, noX, noY, -1, -1, top, 2, nullptr);
        //std::cerr << "Successfully initialize root" << std::endl;
        while(clock() - start_time <= 2.0 * CLOCKS_PER_SEC) {
            Node* v = TreePolicy(root);
            int delta = DefaultPolicy(v);
            BackUp(v, delta);
        }
        Node* choice = root->best_child(0);
        return choice->lastY;
    }
};
