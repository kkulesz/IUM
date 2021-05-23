log_file = "ab_event.log"


def read_data_from_log_file():
    file = open(log_file, "w")


if __name__ == "__main__":
    # TODO: odczytac dane z pliku, podzielic pomiedzy modele i policzyc procent dobrych odpowiedzi w zaleznosci od modelu
    data = read_data_from_log_file()
