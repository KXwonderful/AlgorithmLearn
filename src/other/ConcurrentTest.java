package other;


/**
 * 线程
 */
public class ConcurrentTest {

    public static void main(String[] args) {

        Thread thread1 = new Thread(new Runnable1());
        Thread thread2 = new Thread(new Runnable2());
        //thread1.start();
        //thread2.start();


        // 一个线程需要等待另一个线程执行完才能继续的场景
        // join 向当前线程插入一条任务
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("run 1: " + System.currentTimeMillis());
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("run 2: " + System.currentTimeMillis());
            }
        });

        thread.start();
        try {
            thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("test 3: " + System.currentTimeMillis());
    }

    final String TAG = "ConcurrentTest -->";

    static boolean hasNotify = false;


    // notify 适用于多线程同步，一个线程需要等待另一个线程的结果，或者部分结果
    // 注：要保证 wait、notify 的优先级，即调用顺序
    static final Object object = new Object();

    static class Runnable1 implements Runnable {
        @Override
        public void run() {
            System.out.println("run: thread1 start");
            synchronized (object) {
                try {
                    if (!hasNotify) {
                        object.wait(1000);
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            System.out.println("run: thread1 end");
        }
    }

    static class Runnable2 implements Runnable
    {
        @Override
        public void run() {
            System.out.println("run: thread2 start");
            synchronized (object) {
                object.notify();
                hasNotify = true;
            }

            System.out.println("run: thread2 end");
        }
    }
}
