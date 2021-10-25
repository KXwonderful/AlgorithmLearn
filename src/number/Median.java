package number;

import java.util.PriorityQueue;

/**
 * 中位数
 */
public class Median {

    public static void main(String[] args) {
        // test
        MedianFinder obj = new MedianFinder();
        obj.addNum(2);
        obj.addNum(3);
        double median1 = obj.findMedian();
        obj.addNum(4);
        double median2 = obj.findMedian();
        System.out.println("median1=" + median1 + " median2=" + median2);
    }

    /**
     * 数据流的中位数
     * <p>
     * 中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值
     * <p>
     * [2,3,4] 的中位数是 3
     * [2,3] 的中位数是 (2 + 3) / 2 = 2.5
     */
    static class MedianFinder {

        private final PriorityQueue<Integer> large; // 小堆顶（梯形虽然是小顶堆，但其中的元素较大）
        private final PriorityQueue<Integer> small; // 大堆顶（倒三角虽然是大顶堆，但是其中元素较小）

        public MedianFinder() {
            large = new PriorityQueue<>();
            small = new PriorityQueue<>((a, b) -> b - a);
        }

        /**
         * 从数据流中添加一个整数到数据结构中。
         * <p>
         * 不仅要维护large和small的元素个数之差不超过 1，还要维护large堆的堆顶元素要大于等于small堆的堆顶元素
         * <p>
         * 时间复杂度 O(logN)
         */
        public void addNum(int num) {
            // 想要往large里添加元素，不能直接添加，而是要先往small里添加，
            // 然后再把small的堆顶元素加到large中；向small中添加元素同理
            if (small.size() >= large.size()) {
                small.offer(num);
                large.offer(small.poll());
            } else {
                large.offer(num);
                small.offer(large.poll());
            }
        }

        /**
         * 返回目前所有元素的中位数
         * <p>
         * 时间复杂度 O(1)
         */
        public double findMedian() {
            if (large.isEmpty() && small.isEmpty()) {
                return -1;
            }
            // 如果元素不一样多，多的那个堆的堆顶元素就是中位数
            if (large.size() < small.size()) {
                return small.peek();
            } else if (large.size() > small.size()) {
                return large.peek();
            }
            // 如果元素一样多，两个堆堆顶元素的平均数是中位数
            return (large.peek() + small.peek()) / 2.0;
        }
    }
}
