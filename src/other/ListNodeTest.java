package other;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

public class ListNodeTest {
    public static void main(String[] args) {

    }

    /**
     * 合并两个有序链表
     * 双指针技巧
     */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        // 虚拟头结点
        ListNode dummy = new ListNode(-1), p = dummy;
        ListNode p1 = list1, p2 = list2;

        while (p1 != null && p2 != null) {
            // 比较 p1 和 p2 两个指针
            // 将值较小的的节点接到 p 指针
            if (p1.val > p2.val) {
                p.next = p2;
                p2 = p2.next;
            } else {
                p.next = p1;
                p1 = p1.next;
            }
            // p 指针不断前进
            p = p.next;
        }

        if (p1 != null) {
            p.next = p1;
        }

        if (p2 != null) {
            p.next = p2;
        }

        return dummy.next;
    }


    /**
     * 合并 k 个有序链表
     * <p>
     * 用 优先级队列（二叉堆） 数据结构，把链表节点放入一个最小堆，就可以每次获得k个节点中的最小节点
     * <p>
     * 时间复杂度 O(Nlogk)，其中k是链表的条数，N是这些链表的节点总数。
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) return null;
        // 虚拟头节点
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;
        // 优先级队列，最小堆
        PriorityQueue<ListNode> pq = new PriorityQueue<>(lists.length, (a, b) -> (a.val - b.val));
        // 将 k 个链表的头节点加入最小堆
        for (ListNode head : lists) {
            if (head != null) pq.add(head);
        }

        while (!pq.isEmpty()) {
            // 获取最小节点
            ListNode node = pq.poll();
            p.next = node;
            if (node.next != null) {
                pq.add(node.next);
            }
            // p 指针不断前进
            p = p.next;
        }
        return dummy.next;
    }

    /**
     * 单链表的倒数第 k 个节点
     * <p>
     * 先遍历一遍链表算出n的值，然后再遍历链表计算第n - k个节点，需要遍历两次，但面试希望你只需遍历一次的解法
     * <p>
     * 双指针技巧：p1 先走k步，剩下 n-k 步，此时 p2 指向头节点，p1,p2 同时走 n-k 步，完后 p2 就是指向倒数第k个节点
     * <p>
     * 时间复杂度 O(N)
     */
    ListNode findFromEnd(ListNode head, int k) {
        ListNode p1 = head;
        // p1 先走k步
        for (int i = 0; i < k; i++) {
            p1 = p1.next;
        }
        ListNode p2 = head;
        // p1 和 p2 同时走 n - k 步
        while (p1.next != null) {
            p2 = p2.next;
            p1 = p1.next;
        }
        // p2 现在指向第 n - k 个节点
        return p2;
    }

    /**
     * 删除链表的倒数第 N 个结点
     * <p>
     * 虚拟头结点，也是为了防止出现空指针的情况，
     * 比如说链表总共有 5 个节点，题目就让你删除倒数第 5 个节点，也就是第一个节点，
     * 那按照算法逻辑，应该首先找到倒数第 6 个节点。但第一个节点前面已经没有节点了，这就会出错
     * 有了虚拟节点dummy的存在，就避免了这个问题
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // 虚拟头节点
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        // 删除倒数第 n 个，要先找倒数第 n + 1 个节点
        ListNode x = findFromEnd(dummy, n + 1);
        // 删掉倒数第 n 个节点
        x.next = x.next.next;
        return dummy.next;
    }

    /**
     * 单链表的中点
     * <p>
     * 常规方法是先遍历链表计算n，再遍历一次得到第n / 2个节点，也就是中间节点
     * <p>
     * 若只遍历一次，可用「快慢指针」的技巧：每当慢指针slow前进一步，快指针fast就前进两步，当fast走到链表末尾时，slow就指向了链表中点。
     */
    public ListNode middleNode(ListNode head) {
        // 快慢指针初始化指向 head
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    /**
     * 判断链表是否包含环
     */
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            // 快慢指针相遇，说明含有环
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }

    /**
     * 如果链表中含有环，如何计算这个环的起点
     */
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (fast == slow) break;
        }

        if (fast == null || fast.next == null) {
            // fast 遇到空指针说明没有环
            return null;
        }

        // 重新指向头结点
        slow = head;
        // 快慢指针同步前进，相交点就是环起点
        while (slow != fast) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     * 两个链表是否相交
     * <p>
     * 常规方法：用HashSet记录一个链表的所有节点，然后和另一条链表对比，但这就需要额外的空间
     * <p>
     * 用双指针技巧：让p1遍历完链表A之后开始遍历链表B，让p2遍历完链表B之后开始遍历链表A，这样相当于「逻辑上」两条链表接在了一起，
     * 这样进行拼接，就可以让p1和p2同时进入公共部分
     * 空间复杂度为O(1)，时间复杂度为O(N)
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        // p1 指向 A 链表头结点，p2 指向 B 链表头结点
        ListNode p1 = headA, p2 = headB;
        while (p1 != p2) {
            // p1 走一步，如果走到 A 链表末尾，转到 B 链表
            if (p1 == null) {
                p1 = headB;
            } else {
                p1 = p1.next;
            }
            // p2 走一步，如果走到 B 链表末尾，转到 A 链表
            if (p2 == null) {
                p2 = headA;
            } else {
                p2 = p2.next;
            }
        }
        return p1;
    }

    /**
     * 138. 复制带随机指针的链表
     */
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        // 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        while (cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        // 构建新链表的 next 和 random 指向
        while (cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        // 返回新链表的头节点
        return map.get(head);
    }

    /**
     * 复制普通链表
     */
    public Node copyList(Node head) {
        Node cur = head;
        Node dum = new Node(0), pre = dum;
        while (cur != null) {
            Node node = new Node(cur.val);// 复制节点 cur
            pre.next = node;  // 新链表的 前驱节点 -> 当前节点
            cur = cur.next;
            pre = node;       // 保存当前新节点
        }
        return dum.next;
    }


    public static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public static class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }
}
