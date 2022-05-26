package design.decorate;

public class test {
    public static void main(String[] args) {
        Ultraman ultraman = new Ultraman();
        TigaUltraman tigaUltraman = new TigaUltraman(ultraman);
        tigaUltraman.attack();
    }
}
