package tree;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

/**
 * 二叉树
 */
public class BinaryTree {

    /**
     * 输入一棵普通二叉树，返回节点总数
     * <p>
     * 时间复杂度 O(N)
     */
    int countNodes1(TreeNode root) {
        if (root == null) return 0;
        return 1 + countNodes1(root.left) + countNodes1(root.right);
    }

    /**
     * 输入一棵满二叉树，返回节点总数
     * <p>
     * 时间复杂度 O(logN)
     */
    int countNodes2(TreeNode root) {
        int high = 0; // 二叉树高度
        while (root.left != null) {
            high++;
            root = root.left;
        }
        // 节点总数就是 2^h - 1
        return (int) Math.pow(2, high) - 1;
    }

    /**
     * 输入一棵完全二叉树，返回节点总数
     * <p>
     * 时间复杂度 O(logN*logN)
     */
    int countNodes3(TreeNode root) {
        if (root == null) return 0;
        TreeNode l = root, r = root;
        // 记录左右子树高度
        int hl = 0, hr = 0;
        while (l != null) {
            l = l.left;
            hl++;
        }
        while (r != null) {
            r = r.right;
            hr++;
        }
        // 如果左右子树的高度相同，则是一棵满二叉树
        if (hl == hr) {
            return (int) Math.pow(2, hl) - 1;
        }
        // 如果左右高度不同，则按照普通二叉树的逻辑计算
        return 1 + countNodes3(root.left) + countNodes3(root.right);
    }

    // *****************************  二叉树的最近公共祖先  ******************************

    /**
     * 二叉树的最近公共祖先
     * <p>
     * root节点确定了一棵二叉树，p 和 q 是这这棵二叉树上的两个节点，
     * 返回 p 节点和 q 节点的最近公共祖先节点
     */
    TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {

        //情况 1，如果p和q都在以root为根的树中，函数返回的即使p和q的最近公共祖先节点。
        //情况 2，那如果p和q都不在以root为根的树中怎么办呢？函数理所当然地返回null呗。
        //情况 3，那如果p和q只有一个存在于root为根的树中呢？函数就会返回那个节点。

        // base case
        if (root == null) return null;
        if (root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        // 情况1
        if (left != null && right != null) return root;

        // 情况2
        if (left == null && right == null) return null;
        // 情况3
        return left == null ? right : left;
    }

    /**
     * 二叉树的镜像（翻转二叉树）
     */
    public TreeNode invertTree(TreeNode root) {
        // base case
        if (root == null) return null;
        // root 节点需要交换它的左右子节点
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        // 让左右子节点继续翻转它们的子节点
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    /**
     * 填充二叉树节点的右侧指针
     */
    public TreeNode connect(TreeNode root) {
        if (root == null || root.left == null) return root;
        connectTwoNode(root.left, root.right);
        return root;
    }

    // 将每两个相邻节点都连接起来
    void connectTwoNode(TreeNode node1, TreeNode node2) {
        if (node1 == null || node2 == null) return;
        // 将传入的两个节点连接
        node1.next = node2;

        // 连接相同父节点的两个子节点
        connectTwoNode(node1.left, node1.right);
        connectTwoNode(node2.left, node2.right);
        // 连接跨越父节点的两个子节点
        connectTwoNode(node1.right, node2.left);
    }

    /**
     * 将二叉树展开为链表
     * <p>
     * 定义：将以 root 为根的树拉平为链表
     */
    public void flatten(TreeNode root) {
        // base case
        if (root == null) return;

        flatten(root.left);
        flatten(root.right);

        // 1、左右子树已经被拉平成一条链表
        TreeNode left = root.left;
        TreeNode right = root.right;

        // 2、将左子树作为右子树
        root.left = null;
        root.right = left;

        // 3、将原先的右子树接到当前右子树的末端
        TreeNode p = root;
        while (p.right != null) {
            // 遍历 root 直到 root.right == null
            p = p.right;
        }
        p.right = right;// 接上原先的右子树
    }

    /**
     * 构造最大二叉树
     */
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return build(nums, 0, nums.length - 1);
    }

    // 将 nums[low..high] 构造成符合条件的树，返回根节点
    TreeNode build(int[] nums, int low, int high) {
        // base case
        if (low > high) return null;

        // 找到数组最大值和索引
        int index = -1, maxVal = Integer.MIN_VALUE;
        for (int i = low; i <= high; i++) {
            if (maxVal < nums[i]) {
                maxVal = nums[i];
                index = i;
            }
        }

        TreeNode root = new TreeNode(maxVal);
        // 递归调用构造左右子树
        root.left = build(nums, low, index - 1);
        root.right = build(nums, index + 1, high);

        return root;
    }

    /**
     * 通过前序和中序遍历结果构造二叉树
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return build(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }

    TreeNode build(int[] preorder, int pStart, int pEnd, int[] inorder, int iStart, int iEnd) {
        // base case
        if (pStart > pEnd) return null;

        int rootVal = preorder[pStart];
        int index = 0; // rootVal 在中序遍历的索引
        for (int i = iStart; i < iEnd; i++) {
            if (inorder[i] == rootVal) {
                index = i;
                break;
            }
        }
        TreeNode root = new TreeNode(rootVal);
        int leftSize = index - iStart; // 左子树节点数
        root.left = build(preorder, pStart + 1, pStart + leftSize, inorder, iStart, iStart + leftSize - 1);
        root.right = build(preorder, pStart + leftSize + 1, pEnd, inorder, iStart + leftSize + 1, iEnd);
        return root;
    }

    /**
     * 通过后序和中序遍历结果构造二叉树
     */
    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        return build2(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1);
    }

    TreeNode build2(int[] inorder, int inStart, int inEnd, int[] postorder, int postStart, int postEnd) {
        if (inStart > inEnd) return null;

        int rootVal = postorder[postEnd];
        int index = 0;// rootVal 在中序遍历的索引
        for (int i = inStart; i < inEnd; i++) {
            if (inorder[i] == rootVal) {
                index = i;
                break;
            }
        }

        TreeNode root = new TreeNode(rootVal);
        int leftSize = index - inStart;
        root.left = build2(inorder, inStart, inStart + leftSize - 1, postorder, postStart, postStart + leftSize - 1);
        root.right = build2(inorder, inStart + leftSize + 1, inEnd, postorder, postStart + leftSize, postEnd - 1);
        return root;
    }

    /**
     * 寻找重复子树
     */
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        traverseTree(root);
        return res;
    }

    // 记录所有子树以及出现的次数
    HashMap<String, Integer> memo = new HashMap<>();
    // 记录重复的子树根节点
    LinkedList<TreeNode> res = new LinkedList<>();

    String traverseTree(TreeNode root) {
        if (root == null) return "#"; // 对于空节点，可以用一个特殊字符表示

        // 将左右子树序列化成字符串
        String left = traverseTree(root.left);
        String right = traverseTree(root.right);

        //  左右子树加上自己，就是以自己为根的二叉树序列化结果
        String subTree = left + "," + right + "," + root.val;

        // 获取 subTree 在 memo 中的次数，默认是 0
        int freq = memo.getOrDefault(subTree, 0);
        if (freq == 1) {
            // 多次重复也只会被加入结果集一次
            res.add(root);
        }
        // 给子树对应的出现次数加一
        memo.put(subTree, freq + 1);
        return subTree;
    }

    /**
     * 100.相同的树：给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        // 判断一对节点是否相同
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        if (p.val != q.val) return false;
        // 判断其他节点是否相同
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    /**
     * 101.对称二叉树：给定一个二叉树，检查它是否是镜像对称的
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return isSameVal(root.left, root.right);
    }

    boolean isSameVal(TreeNode left, TreeNode right) {
        // 判断左右子树是否相同
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        return isSameVal(left.left, right.right) && isSameVal(left.right, right.left);
    }

    /**
     * 104.二叉树的最大深度：给定一个二叉树，找出其最大深度
     * <p>
     * 回溯算法
     */
    public int maxDepth(TreeNode root) {
        traverse(root);
        return maxDepth;
    }

    int depth = 0;
    int maxDepth = 0;

    void traverse(TreeNode root) {
        if (root == null) return;

        // 前序遍历
        depth++;
        // 遍历的过程中记录最大深度
        maxDepth = Math.max(maxDepth, depth);
        traverse(root.left);
        traverse(root.right);
        // 后续遍历位置
        depth--;
    }

    /**
     * 104.二叉树的最大深度：给定一个二叉树，找出其最大深度
     * <p>
     * 动态规划
     * <p>
     * 定义：输入一个节点，返回以该节点为根的二叉树的最大深度
     */
    public int maxDepth2(TreeNode root) {
        if (root == null) return 0;
        int leftMax = maxDepth(root.left);
        int rightMax = maxDepth(root.right);
        // 根据左右子树的最大深度推出原二叉树的最大深度
        return 1 + Math.max(leftMax, rightMax);
    }

    /**
     * 112.路径总和：判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum
     */
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        if (root.left == null && root.right == null &&root.val == targetSum) {
            return true;
        }
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, root.val);
    }

    /**
     * 113.路径总和 II：二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
     */
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        traverse(root, targetSum);
        return pathSumRes;
    }

    List<List<Integer>> pathSumRes = new ArrayList<>();
    List<Integer> track = new ArrayList<>();
    int sum = 0;

    void traverse(TreeNode root, int targetSum) {
        if (root == null) return;

        sum += root.val;
        track.add(root.val);
        if (sum == targetSum && root.left == null && root.right == null) {
            pathSumRes.add(new LinkedList<>(track));
        }
        traverse(root.left, targetSum);
        traverse(root.right, targetSum);
        sum -= root.val;
        track.remove(track.size() - 1);
    }


    /**
     * 基本二叉树节点
     */
    static class TreeNode {
        int val;
        TreeNode left, right, next;

        public TreeNode(int val) {
            this.val = val;
        }
    }
}
