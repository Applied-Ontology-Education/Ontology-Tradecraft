from rdflib.namespace import Namespace
from rdflib.term import URIRef


class BFO:
    """
    BFO Namespace

    Generated from: projects/project-4/assignment/src/cco_merged.ttl
    Date: 2025-11-05 10:58:08.720351
    Mode: Property names use labels, comments use local names
    """

    _NS = Namespace("http://purl.obolibrary.org/obo/")

    # Object Properties
    bearerOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000196")  # BFO_0000196
    concretizes: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000059")  # BFO_0000059
    continuantPartOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000176")  # BFO_0000176
    environs: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000183")  # BFO_0000183
    existsAt: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000108")  # BFO_0000108
    firstInstantOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000221")  # BFO_0000221
    genericallyDependsOn: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000084")  # BFO_0000084
    hasContinuantPart: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000178")  # BFO_0000178
    hasFirstInstant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000222")  # BFO_0000222
    hasHistory: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000185")  # BFO_0000185
    hasLastInstant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000224")  # BFO_0000224
    hasMaterialBasis: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000218")  # BFO_0000218
    hasMemberPart: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000115")  # BFO_0000115
    hasOccurrentPart: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000117")  # BFO_0000117
    hasParticipant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000057")  # BFO_0000057
    hasRealization: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000054")  # BFO_0000054
    hasTemporalPart: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000121")  # BFO_0000121
    historyOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000184")  # BFO_0000184
    inheresIn: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000197")  # BFO_0000197
    isCarrierOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000101")  # BFO_0000101
    isConcretizedBy: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000058")  # BFO_0000058
    lastInstantOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000223")  # BFO_0000223
    locatedIn: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000171")  # BFO_0000171
    locationOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000124")  # BFO_0000124
    materialBasisOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000127")  # BFO_0000127
    memberPartOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000129")  # BFO_0000129
    occupiesSpatialRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000210")  # BFO_0000210
    occupiesSpatiotemporalRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000200")  # BFO_0000200
    occupiesTemporalRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000199")  # BFO_0000199
    occurrentPartOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000132")  # BFO_0000132
    occursIn: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000066")  # BFO_0000066
    participatesIn: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000056")  # BFO_0000056
    precededBy: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000062")  # BFO_0000062
    precedes: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000063")  # BFO_0000063
    realizes: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000055")  # BFO_0000055
    spatiallyProjectsOnto: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000216")  # BFO_0000216
    specificallyDependedOnBy: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000194")  # BFO_0000194
    specificallyDependsOn: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000195")  # BFO_0000195
    temporalPartOf: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000139")  # BFO_0000139
    temporallyProjectsOnto: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000153")  # BFO_0000153

    # Classes
    continuant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000002")  # BFO_0000002
    continuantFiatBoundary: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000140")  # BFO_0000140
    disposition: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000016")  # BFO_0000016
    entity: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000001")  # BFO_0000001
    fiatLine: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000142")  # BFO_0000142
    fiatObjectPart: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000024")  # BFO_0000024
    fiatPoint: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000147")  # BFO_0000147
    fiatSurface: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000146")  # BFO_0000146
    function: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000034")  # BFO_0000034
    genericallyDependentContinuant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000031")  # BFO_0000031
    history: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000182")  # BFO_0000182
    immaterialEntity: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000141")  # BFO_0000141
    independentContinuant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000004")  # BFO_0000004
    materialEntity: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000040")  # BFO_0000040
    object: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000030")  # BFO_0000030
    objectAggregate: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000027")  # BFO_0000027
    occurrent: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000003")  # BFO_0000003
    oneDimensionalSpatialRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000026")  # BFO_0000026
    oneDimensionalTemporalRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000038")  # BFO_0000038
    process: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000015")  # BFO_0000015
    processBoundary: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000035")  # BFO_0000035
    processProfile: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000144")  # BFO_0000144
    quality: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000019")  # BFO_0000019
    realizableEntity: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000017")  # BFO_0000017
    relationalQuality: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000145")  # BFO_0000145
    role: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000023")  # BFO_0000023
    site: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000029")  # BFO_0000029
    spatialRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000006")  # BFO_0000006
    spatiotemporalRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000011")  # BFO_0000011
    specificallyDependentContinuant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000020")  # BFO_0000020
    temporalInstant: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000203")  # BFO_0000203
    temporalInterval: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000202")  # BFO_0000202
    temporalRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000008")  # BFO_0000008
    threeDimensionalSpatialRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000028")  # BFO_0000028
    twoDimensionalSpatialRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000009")  # BFO_0000009
    zeroDimensionalSpatialRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000018")  # BFO_0000018
    zeroDimensionalTemporalRegion: URIRef = URIRef("http://purl.obolibrary.org/obo/BFO_0000148")  # BFO_0000148

