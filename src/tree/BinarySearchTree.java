package tree;

import java.util.LinkedList;
import java.util.List;

/**
 * 二叉搜索树
 */
public class BinarySearchTree {

    /**
     * 230. 二叉搜索树中第K小的元素
     */
    public int kthSmallest(TreeNode root, int k) {
        traverse(root, k);
        return res;
    }

    int res = 0;  // 记录结果
    int rank = 0; // 记录当前元素的排名

    // 的中序遍历结果是有序的（升序）
    void traverse(TreeNode root, int k) {
        if (root == null) return;

        traverse(root.left, k);
        rank++;
        if (rank == k) {
            // 找到第 k 小的元素
            res = root.val;
            return;
        }
        traverse(root.right, k);
    }

    /**
     * 538. 1038. BST 转化累加树
     */
    public TreeNode bstToGst(TreeNode root) {
        traverse(root);
        return root;
    }

    int sum = 0;// 记录累加和

    void traverse(TreeNode root) {
        if (root == null) return;

        traverse(root.right);
        sum += root.val; // 维护累加和
        root.val = sum;  // 将 BST 转化成累加树
        traverse(root.left);
    }

    /**
     * 判断 BST 的合法性
     */
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, null, null);
    }

    // 限定以 root 为根的子树节点必须满足 max.val > root.val > min.val
    boolean isValidBST(TreeNode root, TreeNode min, TreeNode max) {
        if (root == null) return true;

        // 若 root.val 不符合 max 和 min 的限制，说明不是合法 BST
        if (min != null && root.val <= min.val) return false;
        if (max != null && root.val >= max.val) return false;

        return isValidBST(root.left, min, root) && isValidBST(root.right, root, max);
    }

    /**
     * 在 BST 中搜索一个数
     */
    public boolean isInBST(TreeNode root, int target) {
        if (root == null) return false;
        if (root.val == target) return true;
        if (root.val < target) return isInBST(root.right, target);
        return isInBST(root.left, target);
    }

    /**
     * 在 BST 中插入一个数
     */
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) return new TreeNode(val);

        // BST 中一般不会插入已存在元素
        if (root.val < val) root.right = insertIntoBST(root.right, val);
        if (root.val > val) root.left = insertIntoBST(root.left, val);
        return root;
    }

    /**
     * 在 BST 中删除一个数
     */
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root.val == key) {
            // 找到了进行删除：
            // 1. A 是末端节点，两个子节点都为空
            if (root.left == null && root.right == null) return null;
            // 2. A 只有一个非空子节点，那么它要让这个子节点接替自己的位置
            if (root.left == null) return root.right;
            if (root.right == null) return root.left;
            // 3. A 有两个子节点，为了不破坏 BST 的性质，A 必须找到左子树中最大的那个节点，或者右子树中最小的那个节点来接替自己
            if (root.right != null) {
                TreeNode minNode = getMin(root.right); // 找到右子树的最小节点
                root.val = minNode.val;                // 把 root 改成 minNode
                root.right = deleteNode(root.right, minNode.val); // 转而去删除 minNode
            }
        } else if (root.val > key) {
            root.left = deleteNode(root.left, key);
        } else if (root.val < key) {
            root.right = deleteNode(root.right, key);
        }

        return root;
    }

    // 获取 BST 最小节点
    TreeNode getMin(TreeNode node) {
        // BST 最左边的就是最小的
        while (node.left != null) {
            node = node.left;
        }
        return node;
    }

    /**
     * 96. 不同的二叉搜索树 I
     */
    public int numTrees(int n) {
        // 计算闭区间 [1, n] 组成的 BST 个数
        memo = new int[n + 1][n + 1];
        return count(1, n);
    }

    int[][] memo;

    // 定义：闭区间 [low, high] 的数字能组成 count(low, high) 种 BST
    int count(int low, int high) {
        // base case
        if (low > high) return 1;

        if (memo[low][high] != 0) return memo[low][high];

        int res = 0;
        for (int i = low; i <= high; i++) {
            // i 的值作为根节点 root
            int left = count(low, i - 1);
            int right = count(i + 1, high);
            // 左右子树的组合数乘积是 BST 的总数
            res += left * right;
        }
        memo[low][high] = res;
        return res;
    }


    /**
     * 95. 不同的二叉搜索树 II
     */
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return new LinkedList<>();
        return build(1, n);
    }

    List<TreeNode> build(int low, int high) {
        List<TreeNode> res = new LinkedList<>();

        // base case
        if (low > high) {
            res.add(null);
            return res;
        }

        // 1、穷举 root 节点的所有可能。
        for (int i = low; i <= high; i++) {
            // 2、递归构造出左右子树的所有合法 BST。
            List<TreeNode> leftTree = build(low, i - 1);
            List<TreeNode> rightTree = build(i + 1, high);
            // 3、给 root 节点穷举所有左右子树的组合。
            for (TreeNode left : leftTree) {
                for (TreeNode right : rightTree) {
                    // i 作为根节点 root 的值
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    res.add(root);
                }
            }
        }
        return res;
    }


    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
}
