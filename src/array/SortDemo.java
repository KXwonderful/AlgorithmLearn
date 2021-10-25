package array;

import java.util.PriorityQueue;
import java.util.Random;

/**
 * 排序算法
 */
public class SortDemo {

    /**
     * 215. 数组中的第K个最大元素
     * <p>
     * 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
     * <p>
     * 注：是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
     */
    public int findKthLargest(int[] nums, int k) {
        return findKthLargest1(nums, k);
    }

    /**
     * 二叉堆解法
     * <p>
     * 二叉堆（优先队列），它会自动排序
     * <p>
     * 时间复杂度 O(NlogK) 空间复杂度 O(K)
     */
    public int findKthLargest1(int[] nums, int k) {
        // 小顶堆，堆顶是最小元素
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int num : nums) {
            // 每个元素都要过一遍二叉堆
            pq.offer(num);
            // 堆中元素多于 k 个时，删除堆顶元素
            if (pq.size() > k) {
                pq.poll();
            }
        }
        // pq 中剩下的是 nums 中 k 个最大元素，堆顶是最小的那个，即第 k 个最大元素
        return pq.peek();
    }

    /**
     * 快速选择算法
     * <p>
     * 快速选择算法比较巧妙，时间复杂度更低，是快速排序的简化版，一定要熟悉思路
     * <p>
     * 时间复杂度 O(N)
     */
    public int findKthLargest2(int[] nums, int k) {
        // 首先为了尽可能防止极端情况发生，随机打乱数组
        shuffle(nums);

        int low = 0, high = nums.length - 1;
        // 索引转化
        k = nums.length - k;
        while (low <= high) {
            // 在 nums[lo..hi] 中选一个分界点
            int p = partition(nums, low, high);

            if (p < k) {
                // 第 k 大的元素在 nums[p+1..hi] 中
                low = p + 1;
            } else if (p > k) {
                // 第 k 大的元素在 nums[lo..p-1] 中
                high = p - 1;
            } else {
                // 找到第 k 大元素
                return nums[p];
            }
        }
        return -1;
    }

    /**
     * 对数组元素进行随机打乱
     */
    public void shuffle(int[] nums) {
        int n = nums.length;
        Random random = new Random();
        for (int i = 0; i < n; i++) {
            // 从 i 到最后随机选一个元素
            int r = i + random.nextInt(n - i);
            swap(nums, i, r);
        }
    }

    public void sort(int[] nums) {
        quickSort(nums, 0, nums.length - 1);
    }

    /**
     * 快速排序
     * <p>
     * 对 nums[lo..hi] 进行排序，先找一个分界点 p，
     * 通过交换元素使得 nums[lo..p-1] 都小于等于 nums[p]，且 nums[p+1..hi] 都大于 nums[p]，
     * 然后递归地去 nums[lo..p-1] 和 nums[p+1..hi] 中寻找新的分界点，最后整个数组就被排序了
     */
    public void quickSort(int[] nums, int low, int high) {
        if (low >= high) return;
        // 通过交换元素构建分界点索引 p
        int p = partition(nums, low, high);
        // 现在 nums[lo..p-1] 都小于 nums[p]，且 nums[p+1..hi] 都大于 nums[p]
        quickSort(nums, low, p - 1);
        quickSort(nums, p + 1, high);
    }

    public int partition(int[] nums, int low, int high) {
        if (low == high) return low;
        // 将 nums[lo] 作为默认分界点 pivot
        int pivot = nums[low];
        // j = hi + 1 因为 while 中会先执行 --
        int i = low, j = high + 1;
        while (true) {
            // 保证 nums[lo..i] 都小于 pivot
            while (nums[++i] < pivot) {
                if (i == high) break;
            }
            // 保证 nums[j..hi] 都大于 pivot
            while (nums[--j] > pivot) {
                if (j == low) break;
            }
            if (i >= j) break;
            // 如果走到这里，一定有：
            // nums[i] > pivot && nums[j] < pivot
            // 所以需要交换 nums[i] 和 nums[j]，
            // 保证 nums[lo..i] < pivot < nums[j..hi]
            swap(nums, i, j);
        }
        // 将 pivot 值交换到正确的位置
        swap(nums, j, low);
        // 现在 nums[lo..j-1] < nums[j] < nums[j+1..hi]
        return j;
    }

    /**
     * 交换数组中的两个元素
     */
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
