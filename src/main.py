from src.nn.keras_nn import _start_nn, _evaluete_nn, _load_data_nn
from src.deep.deep_keras_conv import _start_cnn, _evaluete_cnn, _load_data_cnn
from src.uns_deep.autoencoder_dense import _start_uns


nn_path = './nn'
deep = './deep'
uns_deep = './uns_deep'


def _print_menu():
    print('== MENU Classification OilSpill ==')
    print('----------------------------------')
    print('1 - Neural Network')
    print('2 - Convolutional Neural Network')
    print('3 - Fit and Evaluete Unsupervised Network')
    print()
    a = input('> ')
    if a == '1':
        loop = False
        _print_menu_nn()
    elif a == '2':
        _print_menu_cnn()
    elif a == '3':
        _start_uns()
        _print_menu()


def _print_menu_nn_evaluete():
    print('Seleziona uno dei 10 modelli salvati, digita un valore tra 0 e 9')
    val = input('> ')
    val = int(val)
    X_train, X_test, Y_train, Y_test = _load_data_nn()
    _evaluete_nn(X_train, X_test, Y_train, Y_test, val)
    _print_menu()


def _print_menu_nn():
    print('1 - Fit and Evaluete Neural Network')
    print('2 - Evaluete Neural Network')
    print('3 - back')
    print()
    a = input('> ')
    if a == '1':
        _start_nn()
    elif a == '2':
        _print_menu_nn_evaluete()
    elif a == '3':
        _print_menu()
    _print_menu()


def _print_menu_cnn():
    print('1 - Fit and Evaluete CNN')
    print('2 - Evaluete CNN')
    print('3 - back')
    print()
    a = input('> ')
    if a == '1':
        _start_cnn()
    elif a == '2':
        X_train, X_test, Y_train, Y_test = _load_data_cnn()
        _evaluete_cnn(X_train, X_test, Y_train, Y_test)
        _print_menu()
    elif a == '3':
        _print_menu()
    _print_menu()


# def _print_menu_uns():
#     print('1 - Fit and Evaluete Unsupervised Network')
#     print('2 - Evaluete Unsupervised Network')
#     print('3 - back')
#     print()
#     a = input('> ')
#     if a == '1':
#         _start_uns()
#     elif a == '2':
#         X_train, X_test, Y_train, Y_test = _load_data_cnn()
#         _evaluete_cnn(X_train, X_test, Y_train, Y_test)
#         _print_menu()
#     elif a == '3':
#         _print_menu()
#     _print_menu()


def main():
    print('\n' * 1000)
    _print_menu()


if __name__ == "__main__":
    main()


