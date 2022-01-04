package other;

import java.util.LinkedList;
import java.util.Queue;


public class MyStack {

    Queue<Integer> q = new LinkedList<>();
    int top_elem = 0;

    // 添加元素到栈顶
    public void push(int x) {
        q.offer(x);
        top_elem = x;
    }

    /**
     * 删除栈顶的元素并返回
     * <p>
     * 底层数据结构是先进先出的队列，每次pop只能从队头取元素；
     * 但是栈是后进先出，也就是说popAPI 要从队尾取元素
     * 把队列前面的都取出来再加入队尾，让之前的队尾元素排到队头，这样就可以取出了
     */
    public int pop() {
        int size = q.size();
        while (size > 2) {
            q.offer(q.poll());
            size--;
        }
        // 记录新的队尾元素
        top_elem = q.peek();
        q.offer(q.poll());
        // 之前的队尾元素已经到了队头, 删除之前的队尾元素
        return q.poll();
    }

    // 返回栈顶元素
    public int top() {
        return top_elem;
    }

    // 判断栈是否为空
    public boolean empty() {
        return q.isEmpty();
    }

}
