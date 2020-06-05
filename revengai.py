from document_similarity_computer import document_query
from naive_bayes import naive_bayes
from neuralnetwork_driver import instantiate_neural_network, query_neural_network
from optimized_cosine_distance import optimized_improved_cosine_distance_tfidf, optimized_cosine_distance_tfidf
from reuter_handler import get_topics_from_list_ids, get_content_from_newid
from rocchio import rocchio
from util import get_first_n_instances_from_top_score


def driver():
    #     with open('inverted_index.json', 'r') as infile:
    #         ii = json.load(infile)
    #     with open('dictionary.json', 'r') as infile:
    #         d = json.load(infile)
    #     with open('idf.json', 'r') as infile:
    #         idf = json.load(infile)
    #     with open('inv_tfidf.json', 'r') as infile:
    #         inv_tfidf = json.load(infile)

    query3 = 'The Agricultural Stabilization and Conservation Service (ASCS) has'
    query1 = 'The U.S. Agriculture Department ' \
             'reported the farmer-owned reserve national five-day average ' \
             'price through February 25 as follows (Dlrs/Bu-Sorghum Cwt)'
    query2 = 'the Bahia cocoa zone, alleviating the drought since early ' \
             'January and improving prospects for the coming temporao, ' \
             'although normal humidity levels have not been restored, ' \
             'Comissaria Smith said in its weekly review.'

    # 9
    query9 = 'Champion Products Inc said its ' \
             'board of directors approved a two-for-one stock split of its ' \
             'common shares for shareholders of record as of April 1, 1987.'

    query10 = 'Computer Terminal Systems'

    # 19005
    query19005 = r'Australia''s economy should manage modest ' \
             'growth over the next two years after a sharp slowdown but ' \
             'unemployment could still edge upwards, the Organisation for ' \
             'Economic Cooperation and Development (OECD) said. ' \
             'The organisation''s latest half-yearly report says Gross Domestic ' \
             'Product will grow by 2.5 pct this year and by 2.75 pct ' \
             'in 1988 compared with only 1.4 pct in 1986. The growth will be ' \
             'helped by higher stockbuilding and stronger domestic demand ' \
             'following tax cuts and higher real wages, it added. ' \
             'The report forecasts a decline in inflation, with consumer ' \
             'prices increasing by 8.5 pct this year and 6.25 pct in 1988. ' \
             ' The current account deficit shows signs of easing slightly ' \
             'and could narrow to 12 billion dlrs by the end of 1988. ' \
             'While predicting slightly stronger growth than last year, ' \
             'however, the report revises downwards the OECD''s earlier growth  ' \
             'forecast for 1987 of 3.75 pct. ' \
             'The OECD predicts a similar combination of modest economic ' \
             'growth and rising unemployment for New Zealand, which is ' \
             'struggling to recover from a major economic crisis.'
    # 19006
    query19006 = 'Robert Fildes, president and chief ' \
             'executive of Cetus Corp &lt;CTUS.O>, told Reuters that Squibb Corp ' \
             'is not interested in buying Cetus. ' \
             'Earlier the companies said Squibb would buy from Cetus a ' \
             'five pct equity postion in Cetus for about 40 mln dlrs. ' \
             '"This is not an attempt by Squibb to become a major ' \
             'majority holder in Cetus," Fildes told Reuters in an interview. ' \
             '"Squibb has not approached us with any indication that they ' \
             'want to acquire us and we wouldn''t be interested in that kind  ' \
             'of arrangement," said Fildes. ' \
             'Squibb could not be reached to comment on the late comments ' \
             'by Fildes. ' \
             'Squibb is Cetus'' first pharmaceutical partner and the only  ' \
             'one to own an equity position in Cetus. Eastman Kodak Co &lt;EK> ' \
             'and W.R. Grace &lt;WR> both have joint ventures with Cetus, but ' \
             'neither owns an equity position in the company, said Fildes. ' \
             'Cetus has a venture with Kodak to develp diagnostic ' \
             'products and with Grace to develop agricultural products.'

    # 13003

    # print(compute_similarity(query1, query2, 'tfidf', d, idf))

    # print(document_query(query2, 'hot_encoding', 'cosine_distance'))
    # print(document_query(query2, 'term_frequency', 'cosine_distance'))
    # print(document_query(query2, 'tfidf', 'cosine_distance'))
    top = 5
    encoding_type = 'term_frequency'

    nn = instantiate_neural_network(0.001, load=True)

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
