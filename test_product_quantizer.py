import numpy as np
from product_quantizer import ProductQuantizer

def test_codebook_shapes():
    data = np.random.rand(10, 4)  # 10 vectors, 4D
    pq = ProductQuantizer(num_subvectors=2, num_centroids=4)
    pq.fit(data)
    
    assert len(pq.codebooks) == 2
    for cb in pq.codebooks:
        assert cb.shape == (4, 2)  # 4 centroids per subvector, 2D each

def test_encoding_output_shape():
    data = np.random.rand(10, 8)
    pq = ProductQuantizer(num_subvectors=4, num_centroids=2)
    pq.fit(data)
    codes = pq.encode(data)

    assert codes.shape == (10, 4)
    assert np.all((codes >= 0) & (codes < 2))  # Each index must be in [0, 1]

def test_encoding_is_consistent():
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0]
    ])
    pq = ProductQuantizer(num_subvectors=2, num_centroids=2)
    pq.fit(data)
    codes = pq.encode(data)
    
    # Identical vectors → identical codes
    assert np.array_equal(codes[0], codes[1])

def test_adc_distance_correctness():
    # 4 vectors, 6 dimensions
    data = np.array([
    [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    [9.0, 9.0, 8.0, 8.0, 7.0, 7.0],
    [1.1, 1.1, 2.1, 2.1, 3.1, 3.1],
    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
])

    # Make the split symmetric (6D → 3 subvectors of 2D),
    # but use 4 vectors so output codes.shape = (4, 3) = not square
    pq = ProductQuantizer(num_subvectors=3, num_centroids=5)
    pq.fit(data)
    codes = pq.encode(data)

    query = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    dists = pq.adc_distance(query, codes)

    # Debug print to inspect shape and values
    print("CODES:\n", codes)
    print("DISTANCES:\n", dists)

    # Assertions
    assert codes.shape == (6, 3)        # 4 vectors, 3 codes each
    assert dists.shape == (6,)          # One distance per vector
    sorted_indices = np.argsort(dists)
    assert sorted_indices[0] == 0     # vector 0 should be closest to query
    assert dists[0] < dists[1]
    assert dists[2] < dists[1]          # Close one is closer than far one
    assert dists[2] > 0.0               # Not identical


def test_adc_distance_monotonicity():
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0]
    ])
    pq = ProductQuantizer(num_subvectors=2, num_centroids=2)
    pq.fit(data)
    codes = pq.encode(data)
    
    query = np.array([1.0, 2.0, 3.0, 4.0])
    dists = pq.adc_distance(query, codes)

    assert dists[0] <= dists[1]  # First vector is closer than second
