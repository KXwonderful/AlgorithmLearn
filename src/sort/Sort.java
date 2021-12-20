package sort;

import java.util.Arrays;

public class Sort {

    /**
     * 排序算法的稳定性
     * 定义：
     * 假定在待排序的记录序列中，存在多个具有相同的关键字的记录，若经过排序，这些记录的相对次序保持不变，
     * 即在原序列中，A1=A2，且A1在A2之前，而在排序后的序列中，A1仍在A2之前，则称这种排序算法是稳定的；否则称为不稳定的。
     *
     * 意义：
     * 1、若只是简单的进行数字的排序，那么稳定性将毫无意义。
     * 2、除非要排序的内容是一个复杂对象的多个数字属性，且其原本的初始顺序存在意义，
     * 那么我们需要在二次排序的基础上保持原有排序的意义，才需要使用到稳定性的算法，
     * 例如要排序的内容是一组原本按照价格高低排序的对象，如今需要按照销量高低排序，使用稳定性算法，
     * 可以使得想同销量的对象依旧保持着价格高低的排序展现，只有销量不同的才会重新排序。
     */

    /**
     * 冒泡排序 平均时间复杂度O(n²)，最坏时间复杂度O(n²)，最好时间复杂度O(n)，空间O(1)，稳定
     * 1. 比较相邻的元素。如果第一个比第二个大，就交换他们两个。
     * 2. 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
     * 3. 针对所有的元素重复以上的步骤，除了最后一个。
     * 4. 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
     */
    private void bubbleSort(int[] array) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array.length - i - 1; j++) {
                int temp;
                if (array[j] > array[j + 1]) {
                    temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }
    }

    private void bubbleSort1(int[] array) {
        for (int i = 0; i < array.length - 1; i++) {
            boolean sorted = true;
            for (int j = 0; j < array.length - i - 1; j++) {
                int temp;
                if (array[j] > array[j + 1]) {
                    temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                    sorted = false;
                }
            }
            if (sorted) break;
        }
    }

    private void bubbleSort2(int[] array) {
        int lastExchangeIndex = 0;
        int sortBorder = array.length - 1;
        for (int i = 0; i < array.length - 1; i++) {
            boolean sorted = true;
            for (int j = 0; j < sortBorder; j++) {
                int temp;
                if (array[j] > array[j + 1]) {
                    temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                    sorted = false;
                    lastExchangeIndex = j;
                }
            }
            sortBorder = lastExchangeIndex;
            if (sorted) break;
        }
    }

    /**
     * 简单选择排序 平均时间复杂度O(n²)，最坏时间复杂度O(n²)，最好时间复杂度O(n²)，空间O(1)，不稳定
     * 1. 首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置。
     * 2. 再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
     * 3. 重复第二步，直到所有元素均排序完毕。
     */
    private void selectSort(int[] array) {
        int len = array.length;
        int minIndex, temp;
        for (int i = 0; i < len - 1; i++) {
            minIndex = i;
            for (int j = i + 1; j < len; j++) {
                if (array[j] < array[minIndex]) { // 寻找最小的数
                    minIndex = j; // 将最小数的索引保存
                }
            }
            temp = array[i];
            array[i] = array[minIndex];
            array[minIndex] = temp;
        }
    }


    /**
     * 简单插入排序：平均时间复杂度O(n²)，最坏时间复杂度O(n²)，最好时间复杂度O(n)，空间O(1)，稳定
     * 1. 将第一待排序序列第一个元素看做一个有序序列，把第二个元素到最后一个元素当成是未排序序列。
     * 2. 从头到尾依次扫描未排序序列，将扫描到的每个元素插入有序序列的适当位置。
     * （如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面。）
     */
    private void insertSort(int[] array) {
        int len = array.length;
        for (int i = 1; i < len; i++) {
            int temp = array[i]; // 记录要插入的数据

            // 从已经排序的序列最右边的开始比较，找到比其小的数
            int j = i;
            while (j > 0 && array[j - 1] > temp) {
                array[j] = array[j - 1];
                j--;
            }
            // 存在比其小的数，插入
            if (j != i) {
                array[j] = temp;
            }
        }
    }

    /**
     * 归并排序：采用分治法  平均时间复杂度O(nlogn) ，最坏时间复杂度O(nlogn) ，最好时间复杂度O(nlogn) ，空间O(n)，稳定
     * 1.申请空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列；
     * 2.设定两个指针，最初位置分别为两个已经排序序列的起始位置；
     * 3.比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置；
     * 4.重复步骤 3 直到某一指针达到序列尾；
     * 5.将另一序列剩下的所有元素直接复制到合并序列尾。
     */
    private int[] mergeSort(int[] array) {
        int middle = (int) Math.floor(array.length / 2);
        int[] left = Arrays.copyOfRange(array, 0, middle);
        int[] right = Arrays.copyOfRange(array, middle, array.length);

        return merge(mergeSort(left), mergeSort(right));
    }

    private int[] merge(int[] left, int[] right) {
        int[] res = new int[left.length + right.length];
        int i = 0;
        while (left.length > 0 && right.length > 0) {
            if (left[0] <= right[0]) {
                res[i++] = left[0];
                left = Arrays.copyOfRange(left, 1, left.length);
            } else {
                res[i++] = right[0];
                right = Arrays.copyOfRange(right, 1, right.length);
            }
        }

        while (left.length > 0) {
            res[i++] = left[0];
            left = Arrays.copyOfRange(left, 1, left.length);
        }

        while (right.length > 0) {
            res[i++] = right[0];
            right = Arrays.copyOfRange(right, 1, right.length);
        }

        return res;
    }


    /**
     * 快速排序：平均时间复杂度O(nlogn) ，最坏时间复杂度O(n²) ，最好时间复杂度O(nlogn) ，空间O(logn)，不稳定
     * 1.从数列中挑出一个元素，称为 "基准"（pivot）;
     * 2.重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。
     * 在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
     * 3.递归地把小于基准值元素的子数列和大于基准值元素的子数列排序；
     */
    private int[] quickSort(int[] array, int left, int right) {
        if (left < right) {
            int partitionIndex = partition(array, left, right);
            quickSort(array, left, partitionIndex - 1);
            quickSort(array, partitionIndex + 1, right);
        }
        return array;
    }

    // 分治（单边循环法）确定基准元素
    private int partition(int[] arr, int left, int right) {
        // 设定基准值
        int pivot = left;
        int index = pivot + 1;
        for (int i = index; i <= right; i++) {
            if (arr[i] < arr[pivot]) {
                swap(arr, i, index);
                index++;
            }
        }
        swap(arr, pivot, index - 1);
        return index - 1;
    }

    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    /**
     * 堆排序：平均时间复杂度O(nlogn) ，最坏时间复杂度O(nlogn) ，最好时间复杂度O(nlogn) ，空间O(1)，不稳定
     * <p>
     * 堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。
     * 堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点.
     * <p>
     * 大顶堆：每个节点的值都大于或等于其子节点的值，在堆排序算法中用于升序排列；
     * 小顶堆：每个节点的值都小于或等于其子节点的值，在堆排序算法中用于降序排列；
     * <p>
     * 1.创建一个堆 H[0……n-1]；
     * 2.把堆首（最大值）和堆尾互换；
     * 3.把堆的尺寸缩小 1，并调用 shift_down(0)，目的是把新的数组顶端数据调整到相应位置；
     * 4.重复步骤 2，直到堆的尺寸为 1
     */
    private void heapSort(int[] array) {
        int len = array.length;
        buildMapHeap(array, len);
        for (int i = len - 1; i > 0; i--) {
            swap(array, 0, i);
            len--;
            heapify(array, 0, len);
        }
    }

    private void buildMapHeap(int[] arr, int len) {
        for (int i = (int) Math.floor(len / 2); i >= 0; i--) {
            heapify(arr, i, len);
        }
    }

    // 堆化
    private void heapify(int[] arr, int i, int len) {
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int largest = i;

        if (left < len && arr[left] > arr[largest]) {
            largest = left;
        }

        if (right < len && arr[right] > arr[largest]) {
            largest = right;
        }

        if (largest != i) {
            swap(arr, i, largest);
            heapify(arr, largest, len);
        }
    }

    /**
     * 计数排序：平均时间复杂度O(n+k) ，最坏时间复杂度O(n+k) ，最好时间复杂度O(n+k) ，空间O(n+k)，稳定
     * <p>
     * 核心是将输入的数据值转化为键存储在额外开辟的数组空间中。
     * <p>
     * 1.找出待排序的数组中最大和最小的元素
     * 2.统计数组中每个值为i的元素出现的次数，存入数组C的第i项
     * 3.对所有的计数累加（从C中的第一个元素开始，每一项和前一项相加）
     * 4.反向填充目标数组：将每个元素i放在新数组的第C(i)项，每放一个元素就将C(i)减去1
     */
    private void countingSort(int[] array) {
        int maxValue = getMaxValue(array);

        int bucketLen = maxValue + 1; // 计数桶长度
        int[] bucket = new int[bucketLen];
        // 初始化计数桶
        for (int value : array) {
            bucket[value]++;
        }

        int sortedIndex = 0;
        for (int j = 0; j < bucketLen; j++) {
            while (bucket[j] > 0) {
                array[sortedIndex++] = j;
                bucket[j]--;
            }
        }
    }

    private int getMaxValue(int[] arr) {
        int maxValue = arr[0];
        for (int value : arr) {
            if (maxValue < value) {
                maxValue = value;
            }
        }
        return maxValue;
    }

    /**
     * 桶排序：平均时间复杂度O(n+k) ，最坏时间复杂度O(n²) ，最好时间复杂度O(n) ，空间O(n+k)，稳定
     * <p>
     * 计数排序的升级版。
     * 它利用了函数的映射关系，高效与否的关键就在于这个映射函数的确定.
     * <p>
     * 1.在额外空间充足的情况下，尽量增大桶的数量
     * 2.使用的映射函数能够将输入的 N 个数据均匀的分配到 K 个桶中
     */
    private void bucketSort(int[] array) {
        int bucketSize = 5;

        // 寻找最大最小数
        int minValue = array[0];
        int maxValue = array[0];
        for (int value : array) {
            if (value < minValue) {
                maxValue = value;
            } else if (value > maxValue) {
                maxValue = value;
            }
        }

        // 初始化桶
        int bucketCount = (int) Math.floor((maxValue - maxValue) / bucketSize) + 1;
        int[][] buckets = new int[bucketCount][0];

        // 利用映射函数将数据分配到各个桶中
        for (int i = 0; i < array.length; i++) {
            int index = (int) Math.floor((array[i] - minValue) / bucketSize);
            buckets[index] = arrAppend(buckets[index], array[i]);
        }

        int arrIndex = 0;
        for (int[] bucket : buckets) {
            if (bucket.length <= 0) continue;
            // 对每个桶进行排序，这边排序可用前面介绍过的排序算法
            insertSort(bucket);
            for (int value : bucket) {
                array[arrIndex++] = value;
            }
        }
    }

    /**
     * 自动扩容，并保存数据
     */
    private int[] arrAppend(int[] arr, int value) {
        arr = Arrays.copyOf(arr, arr.length + 1);
        arr[arr.length - 1] = value;
        return arr;
    }

}
