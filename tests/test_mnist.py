from datasets import package


def test_mnist():
    dp = package.datasets['mnist']
    train, test = dp.make(1000, 100)
    x = train[0]
    print(x.shape)