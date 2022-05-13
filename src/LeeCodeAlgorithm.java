import java.util.Comparator;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * HOT 100
 */
public class LeeCodeAlgorithm {

    public static void main(String[] args) {
        ListNode node1 = new ListNode(1);
        ListNode t = node1;
        t.next = new ListNode(4);
        t = t.next;
        t.next = new ListNode(5);

        ListNode node2 = new ListNode(1);
        ListNode t2 = node2;
        t2.next = new ListNode(3);
        t2 = t2.next;
        t2.next = new ListNode(4);

    }

    /**
     * 23. 合并K个升序链表
     * <p>
     * 给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。
     */
    public static ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        PriorityQueue<ListNode> pq = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
        for (ListNode node : lists) {
            if (node != null) pq.add(node);
        }
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            p.next = node;
            if (node.next != null) {
                pq.add(node.next);
            }
            p = p.next;
        }
        return dummy.next;
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
}
