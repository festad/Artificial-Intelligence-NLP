from document_similarity_computer import document_query
from naive_bayes import naive_bayes
from neuralnetwork_driver import instantiate_neural_network, query_neural_network
from reuter_handler import get_topics_from_list_ids, get_content_from_newid
from rocchio import rocchio
from util import get_first_n_instances_from_top_score


def driver():
    top = 5
    encoding_type = 'term_frequency'

    while(True):

        print('1 -> Query')
        print('2 -> Document ID')
        print('-> ')
        choice = int(input())
        if choice == 1:
            print('Insert a query:\n-> ')
            query = input()
        elif choice == 2:
            print('Insert an id: \n-> '),
            newid = input()
            query = get_content_from_newid(newid)
            print(query[:200] + '...')
        else:
            print('Bye')
            exit(-1)

        print("")

        print('1 -> Naive Bayes')
        print('2 -> Cosine distance')
        print('3 -> Rocchio')
        print('4 -> Neural network')
        print('-> ')
        algorithm = int(input())
        if algorithm == 2 or algorithm == 3:
            print('1 -> hot encoding')
            print('2 -> term frequency')
            print('3 -> tfidf')
            print('-> ')
            encoding = int(input())
            if encoding == 1:
                encoding_type = 'hot_encoding'
            elif encoding == 2:
                encoding_type = 'term_frequency'
            elif encoding == 3:
                encoding_type = 'tfidf'

        print("")

        # top_documents_ids = optimized_cosine_distance_tfidf(query)
        # print("optimized_cosine_distance with tfidf -> " + str(
        #     get_first_n_instances_from_top_score(top_documents_ids, top)))
        # print("topics from optimized cosine distance with tfidf -> " + str(
        #     get_topics_from_list_ids(get_first_n_instances_from_top_score(top_documents_ids, top))))

        # imp_top_documents_ids = optimized_improved_cosine_distance_tfidf(query)
        # print("optimized_improved_cosine_distance with tfidf -> " + str(
        #     get_first_n_instances_from_top_score(imp_top_documents_ids, top)))
        # print("topics from improved optimized cosine distance with tfidf -> " + str(
        #     get_topics_from_list_ids(get_first_n_instances_from_top_score(imp_top_documents_ids, top))))


        if algorithm == 1:
            print("naive bayes -> " + str(get_first_n_instances_from_top_score(naive_bayes(query), top)))

        elif algorithm == 2:
            top_non_optimized_ids = document_query(query, encoding_type)
            print("cosine distance with encoding: " + encoding_type + " -> " + str(
                get_first_n_instances_from_top_score(top_non_optimized_ids, top)))
            print("topics from cosine distance with encoding " + encoding_type + " -> " + str(
                get_topics_from_list_ids(get_first_n_instances_from_top_score(top_non_optimized_ids, top))))

        elif algorithm == 3:
            print("rocchio with encoding: " + encoding_type + " -> " + str(
                get_first_n_instances_from_top_score(rocchio(query, encoding_type), top)))

        elif algorithm == 4:
            nn = instantiate_neural_network(0.001, load=True)
            print("neural network with tfidf: -> " + str(query_neural_network(nn, query)))


        # top_non_optimized_ids = document_query(query, encoding_type)
        # print("cosine distance with encoding: " + encoding_type + " -> " + str(
        #     get_first_n_instances_from_top_score(top_non_optimized_ids, top)))
        # print("topics from cosine distance with encoding " + encoding_type + " -> " + str(
        #     get_topics_from_list_ids(get_first_n_instances_from_top_score(top_non_optimized_ids, top))))
        #
        # print("rocchio with encoding: " + encoding_type + " -> " + str(
        #     get_first_n_instances_from_top_score(rocchio(query, encoding_type), top)))
        #
        # print("naive bayes -> " + str(get_first_n_instances_from_top_score(naive_bayes(query), top)))

        # nn = instantiate_neural_network(0.001, load=True)
        # print("neural network with tfidf: -> " + str(query_neural_network(nn, query)))
        print("")


if __name__ == '__main__':
    driver()
