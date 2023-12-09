from datasketch import MinHash

data = ["""
In Frank Lloyd Wright's vision of Broadacre City, he championed a return to agrarian lifestyles and advocated for the dissolution of the dichotomy between urban and rural areas. He criticized what he perceived as the excesses of "over-built cities of old" (Wright 1935). In contrast, Le Corbusier advocated for increased urbanization through communal living spaces characterized by large, shared environments accommodating a multitude of residents.

WORKac's proposal, Nature City, ostensibly aims to reconcile these two contrasting visions by blending the conveniences of urban life with the health benefits associated with country living. However, a closer examination reveals a tendency towards urbanization and centralization, aligning more with Le Corbusier's ideals than Wright's. Wright would likely critique Nature City for its high density and overbuilding, as it deviates from his vision of integrating dwellings into rural landscapes.

Nature City, despite being surrounded by nature, diverges significantly from Wright's concept of a city harmonizing with nature. It resembles a modern city encircled by suburbs, featuring multiple high-rises, public transportation systems, and a densely centralized city center. This stands in stark contrast to Broadacre City, where the emphasis is on farm living, decentralization, and individual motor vehicles for transportation. Wright valued the beauty derived from houses immersed in natural surroundings, whereas Nature City contains nature within an urban framework.

Wright's insistence on uniformity and lack of distinction in housing quality contrasts sharply with the varied housing options and affordability levels present in Nature City. Wright would likely perceive the diversity in housing types as a form of regimentation and an impediment to the essence of life, given his belief that any form of regimentation equates to a stifling uniformity and, ultimately, death.

Examining Nature City through Le Corbusier's lens provides a different perspective. Le Corbusier, known for his Dom-Ino house principle and stacked units, would likely appreciate the interconnected and stacked nature of the buildings in Nature City. The megastructure resembling Le Corbusier's Unite d'Habitation, with its emphasis on shared living spaces, aligns with his ideals of urban fabric raised on columns and interconnected residences.

In conclusion, while Frank Lloyd Wright would vehemently criticize Nature City for its departure from his decentralized, nature-centric vision, Le Corbusier would likely support it for its adherence to principles of urbanization and communal living spaces that align with his architectural ideals. The tension between these contrasting visions underscores the complexity of envisioning and implementing sustainable and livable urban environments.

""", 
"""
In Frank Lloyd Wright's envisioning of Broadacre City, he ardently championed a return to agrarian lifestyles and advocated for the dissolution of the dichotomy between urban and rural areas. Criticizing what he perceived as excesses in "over-built cities of old" (Wright 1935), Wright's vision aimed at integrating dwellings into rural landscapes, emphasizing decentralization and individual motor vehicles for transportation.

On the contrary, Le Corbusier advocated for increased urbanization through communal living spaces characterized by large, shared environments accommodating numerous residents. His vision focused on high-density, interconnected residences, contrasting sharply with Wright's ideals.

WORKac's proposal, Nature City, seemingly aspires to reconcile these opposing visions by blending urban conveniences with the health benefits associated with country living. However, a closer examination reveals a tendency towards urbanization and centralization, aligning more with Le Corbusier's ideals than Wright's. Wright, who valued the integration of houses into natural surroundings, would likely criticize Nature City for its high density, overbuilding, and deviation from his vision.

Nature City, surrounded by nature, diverges significantly from Wright's concept of a city harmonizing with nature. It resembles a modern city encircled by suburbs, featuring high-rises, public transportation systems, and a densely centralized city center. This contrasts starkly with Broadacre City's emphasis on farm living, decentralization, and individual vehicles.

Wright's insistence on uniformity and lack of distinction in housing quality contrasts sharply with the varied housing options and affordability levels present in Nature City. He might perceive the diversity in housing types as a form of regimentation, contrary to his belief that regimentation stifles uniformity and, ultimately, life.

Examining Nature City through Le Corbusier's lens provides a different perspective. Le Corbusier, known for his Dom-Ino house principle and stacked units, might appreciate the interconnected and stacked nature of the buildings in Nature City. The megastructure resembling Le Corbusier's Unite d'Habitation aligns with his ideals of urban fabric raised on columns and interconnected residences.

In conclusion, while Frank Lloyd Wright would likely criticize Nature City for its departure from his decentralized, nature-centric vision, Le Corbusier might support it for its adherence to principles of urbanization and communal living spaces. The tension between these contrasting visions underscores the complexity of envisioning and implementing sustainable and livable urban environments.

""", 
"""
Frank Lloyd Wright's vision of Broadacre City, advocating for a return to agrarian lifestyles and the dissolution of the urban-rural dichotomy, presents a nostalgic perspective that might overlook the benefits of urbanization. In contrast, Le Corbusier's emphasis on increased urbanization and communal living spaces reflects a more pragmatic approach that acknowledges the efficiencies of centralized structures. Examining WORKac's proposal, Nature City, reveals its potential merits in embracing urbanization and shared environments, aligning with the principles championed by Le Corbusier.

While Wright criticized the "over-built cities of old" and championed decentralization, Nature City represents a departure from his ideals. However, this departure may be viewed as progress, as urbanization brings about enhanced infrastructure, efficient public transportation systems, and economies of scale. The densely centralized city center in Nature City could be seen as a practical response to the challenges of modern living, catering to the growing population and fostering a sense of community.

Nature City, surrounded by nature but resembling a modern city with high-rises and diverse housing options, challenges the notion of integrating dwellings into rural landscapes. Nevertheless, this departure from Wright's vision can be seen as an evolution towards a more sustainable and efficient urban model. The incorporation of nature within an urban framework may provide residents with the best of both worlds – the conveniences of city life alongside the therapeutic benefits of natural surroundings.

Wright's insistence on uniformity in housing quality might be perceived as restrictive, limiting individual expression and choice. Nature City's diverse housing options and affordability levels, on the other hand, align with the principles of choice and flexibility. The varied housing types can be viewed as a response to the diverse needs and preferences of a modern, heterogeneous society, promoting inclusivity and catering to a wide range of residents.

Examining Nature City through Le Corbusier's lens reveals its potential alignment with his architectural ideals. The interconnected and stacked nature of the buildings, reminiscent of Le Corbusier's Dom-Ino house principle and megastructures, may be appreciated for its efficient land use and resource optimization. The emphasis on shared living spaces and a densely interconnected urban fabric reflects the practicality of addressing the challenges posed by rapid urbanization.

In conclusion, while Frank Lloyd Wright's vision of Broadacre City has its merits, it is essential to recognize the potential advantages of urbanization and centralized living spaces. Nature City, despite deviating from Wright's ideals, presents a modern and pragmatic approach that aligns with the principles advocated by Le Corbusier. The tension between these contrasting visions highlights the ongoing debate about the most effective and sustainable urban models for the future.
"""]

test = """"
In Frank Lloyd Wright's Broadacre City, Wright emphasized the need for a return to the life of a farmer; he called for the abolition of the opposition of town and country and pushed back against what he called the “over-built cities of old” (Wright 1935). On the other hand, Le Corbusier advocated for increased urbanization through communal living space; large, shared spaces occupied by many people. 

Nature City, WORKac's proposal for a “sustainable” city in Keizer, while claiming to bridge the gap between those two ideals - combining “the convinces of urban life with the health benefits… of country living” - tends towards urbanization and centralization, over rural living, and decentralization. Therefore, Wright would vehemently criticize Nature City due to its high density and over-building; while Le Corbusier would support Nature City as it fits perfectly into his vision for highly dense urban fabric and his personal views on domesticity and family living.

Nature City, while surrounded by nature, is anything but a city belonging to nature. It is a development, akin to a modern city surrounded by suburbs, with multiple high rises, a system of public transportation and a highly dense, highly centralized city center. This is diametrically opposed to Wright's vision for integrating dwellings into rural life. Indeed, Broadacre City is fundamentally built on farm living, where the “farm itself… becomes the most attractive unit of the city” (Wright 1935). This inherently produces decentralization, where the farms are surrounded by sparsely populated prairie houses, and transportation revolves around the notion of the individual motor vehicle; not fixed transportation, which Wright restricts only for long-distance transportation. For Wright, beauty comes from the houses being surrounded by nature, having an unobstructed view of the farmland, and not from nature contained within urbanity such as what is seen in Nature City. Moreover, in Broadacres there is “no distinction exists between much and little, more and less. Quality is in all, for all, alike” (Wright 1935); every dwelling is the same in terms of quality and only differs in individuality. In Nature City, there is a variety of housing types varying in affordability. This along with all other contradictions, Wright would've viewed Nature City as impeding life, as he believed all regimentation is a form of death. 

Evaluating Nature City from Le Corbusier's perspective, however, leads to significantly different conclusions. One of Le Corbusier's key principles was the idea of stacked units based on the Dom-Ino house principle. The idea of an urban fabric raised on columns is apparent in Nature City; Most buildings are interconnected with one another and stacked on top of each other. The largest building resembles a megastructure, and while residential in nature, has a key emphasis on shared living space through the waterfall running down the middle of the building, connecting all the residences to one another, and centralizing at the base of the building. Indeed, this is reminisced of Le Corbusier's Unite d'Habitation; a city within a city. Thus, by drawing parallels with Le Corbusier's ideals and Nature City, it becomes evident that Le Corbusier would be in support of the proposal
rewrite this essay
"""

test2 = """"
Frank Lloyd Wright's vision of Broadacre City, advocating for a return to agrarian lifestyles and the dissolution of the urban-rural dichotomy, offers a nostalgic perspective that may undervalue the potential advantages of urbanization. In contrast, Le Corbusier's emphasis on increased urbanization and communal living spaces reflects a pragmatic approach that acknowledges the efficiencies of centralized structures. Examining WORKac's proposal, Nature City, reveals its potential merits in embracing urbanization and shared environments, aligning with the principles championed by Le Corbusier.

Wright's critique of "over-built cities of old" and his advocacy for decentralization may be reconsidered in the context of Nature City. This departure from Wright's ideals could be seen as a step forward, introducing enhanced infrastructure, efficient public transportation systems, and economies of scale. The densely centralized city center in Nature City might be interpreted as a practical response to the challenges of modern living, accommodating a growing population while fostering a sense of community.

Nature City, surrounded by nature but resembling a modern city with high-rises and diverse housing options, challenges the notion of integrating dwellings into rural landscapes. However, this departure from Wright's vision may be viewed as an evolution toward a more sustainable and efficient urban model. The incorporation of nature within an urban framework offers residents the benefits of city life alongside the therapeutic advantages of natural surroundings.

Wright's emphasis on uniformity in housing quality could be seen as restrictive, limiting individual expression and choice. Nature City's provision of diverse housing options and affordability levels aligns with the principles of choice and flexibility. The variety in housing types responds to the diverse needs and preferences of a modern, heterogeneous society, promoting inclusivity and catering to a wide range of residents.

Examining Nature City through Le Corbusier's lens highlights its potential alignment with his architectural ideals. The interconnected and stacked nature of the buildings, reminiscent of Le Corbusier's Dom-Ino house principle and megastructures, may be appreciated for its efficient land use and resource optimization. The emphasis on shared living spaces and a densely interconnected urban fabric reflects the practicality of addressing the challenges posed by rapid urbanization.

In conclusion, while Frank Lloyd Wright's vision of Broadacre City has its merits, it is crucial to recognize the potential advantages of urbanization and centralized living spaces. Nature City, despite deviating from Wright's ideals, presents a modern and pragmatic approach that aligns with the principles advocated by Le Corbusier. The tension between these contrasting visions underscores the ongoing debate about the most effective and sustainable urban models for the future.
"""

class DetectAIContent():
    def __init__(self, ai_strings):
        self.k = 10
        self.p = 0.35

        self.ai_minhashes = self.preprocess_content(ai_strings, self.k)
        self.t = self.pairwise_jaccard_similarity(self.ai_minhashes)

        self.actual_t = self.p * self.t

    def classify(self, string):
        minhash = self.compute_minhash(string, k=self.k)
        similarity = self.average_jaccard_similarity(minhash)

        if similarity < self.actual_t:
            return False
        else:
            return True

    def average_jaccard_similarity(self, minhash):
        similarities = [self.calculate_jaccard_similarity(minhash, ai_minhash) for ai_minhash in self.ai_minhashes]
        average_similarity = sum(similarities) / len(similarities)

        return average_similarity

    def compute_minhash(self, text, num_perm=128, k=10):
        minhash = MinHash(num_perm=num_perm)
        # Generate k-shingles
        shingles = set()
        for i in range(len(text) - k + 1):
            shingle = text[i:i+k]
            shingles.add(shingle)

        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))

        return minhash

    def preprocess_content(self, strings, k=10):
        # Compute MinHash for each abstract using K-shingling
        return [self.compute_minhash(string, k=k) for string in strings]
        
    def calculate_jaccard_similarity(self, minhash1, minhash2):
        # Compute the Jaccard similarity between two MinHash objects
        return minhash1.jaccard(minhash2)

    def pairwise_jaccard_similarity(self, minhash_list):
        # Compute pairwise Jaccard similarities
        pairwise_similarities = []
        for i in range(len(minhash_list)):
            for j in range(i + 1, len(minhash_list)):
                similarity = self.calculate_jaccard_similarity(minhash_list[i], minhash_list[j])
                pairwise_similarities.append(similarity)

        # Calculate the average similarity
        average_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
        return average_similarity


def main():
    detector = DetectAIContent(data)
    print(detector.classify(test2))
   

if __name__ == "__main__":
    main()

