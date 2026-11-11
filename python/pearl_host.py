"""Torch-free host-side Pearl commitment + proof building (numpy + pearl_mining).

Mirrors pearl_miner.commitment_hashes / build_proof / verify_proof_local but takes
numpy int8 matrices instead of torch tensors. Validated bit-exact against the GPU
tensor_hash commitment.
"""
from __future__ import annotations

import blake3
import numpy as np

import pearl_mining as pm


def _tree(matrix_i8: np.ndarray):
    """MerkleTree over the chunk-padded matrix bytes (caller binds the key)."""
    return matrix_i8.view(np.uint8).tobytes()


def _merkle_tree(matrix_i8: np.ndarray, key: bytes):
    padded = pm.pad_to_chunk_boundary(np.ascontiguousarray(matrix_i8).view(np.uint8).tobytes())
    return pm.MerkleTree(padded, key)


def commitment_hashes(A: np.ndarray, B: np.ndarray, key: bytes) -> tuple[bytes, bytes]:
    """(noise_seed_A, noise_seed_B). A:[m,k] int8, B:[k,n] int8. Commits A and B^T."""
    root_A = bytes(_merkle_tree(A, key).root)
    root_B = bytes(_merkle_tree(np.ascontiguousarray(B.T), key).root)
    commitment_B = blake3.blake3(key + root_B).digest()
    commitment_A = blake3.blake3(commitment_B + root_A).digest()
    return commitment_A, commitment_B



# Pearl certificate V3 salted noise-seed derivation.
# Consensus definition:
#   salted_A = BLAKE3(root_A || LE32(m) || 28*0, key=SEED_SALT_A)
#   salted_B = BLAKE3(root_B || LE32(n) || 28*0, key=SEED_SALT_B)
#   noise_B  = BLAKE3(job_key || salted_B)
#   noise_A  = BLAKE3(noise_B || salted_A)

_SEED_SALT_A = bytes([
    0x82,0x49,0x40,0x6c,0xa0,0xed,0x15,0x16,
    0x96,0x16,0xf6,0x92,0xfc,0xf0,0x76,0xf8,
    0x92,0xdb,0xdb,0x2a,0x70,0x23,0xb8,0x52,
    0xf0,0xd4,0x77,0x19,0xc3,0x90,0x01,0x7b
])

_SEED_SALT_B = bytes([
    0x11,0x30,0x06,0x32,0xec,0x63,0x01,0xca,
    0x2b,0xe2,0xaf,0x71,0x8b,0x3f,0x4d,0x4f,
    0x1a,0xe9,0xc6,0x39,0x88,0xe8,0xcc,0x04,
    0x48,0x44,0x30,0x1d,0x71,0xb8,0x9a,0xa9
])


def _v3_bind_root(root: bytes, dim: int, salt: bytes) -> bytes:
    msg = root + int(dim).to_bytes(4, "little") + bytes(28)
    assert len(msg) == 64
    return blake3.blake3(msg, key=salt).digest()


def commitment_hashes_bt_v3(
    A: np.ndarray,
    Bt: np.ndarray,
    key: bytes,
) -> tuple[bytes, bytes]:
    """Return (noise_seed_A, noise_seed_B) using Pearl certificate-V3 rules.

    A  : [m,k]
    Bt : [n,k]
    """
    m, k = A.shape
    n, k2 = Bt.shape
    assert k == k2

    raw_root_A = bytes(_merkle_tree(A, key).root)
    raw_root_B = bytes(_merkle_tree(Bt, key).root)

    hash_A = _v3_bind_root(raw_root_A, m, _SEED_SALT_A)
    hash_B = _v3_bind_root(raw_root_B, n, _SEED_SALT_B)

    noise_seed_B = blake3.blake3(key + hash_B).digest()
    noise_seed_A = blake3.blake3(noise_seed_B + hash_A).digest()

    return noise_seed_A, noise_seed_B


def build_proof(A: np.ndarray, B: np.ndarray, winning_row: int, winning_col: int,
                key: bytes, noise_rank: int):
    """Consensus-valid PlainProof for a winning 16x16 tile (host numpy version)."""
    m, k = A.shape
    n = B.shape[1]
    TILE = 16
    a_rows = list(range(winning_row, min(winning_row + TILE, m)))
    b_cols = list(range(winning_col, min(winning_col + TILE, n)))
    tree_A = _merkle_tree(A, key)
    tree_B = _merkle_tree(np.ascontiguousarray(B.T), key)
    li_A = pm.MerkleTree.compute_leaf_indices_from_rows(a_rows, (m, k))
    li_B = pm.MerkleTree.compute_leaf_indices_from_rows(b_cols, (n, k))
    mp_A = pm.MatrixMerkleProof(tree_A.get_multileaf_proof(li_A), a_rows)
    mp_B = pm.MatrixMerkleProof(tree_B.get_multileaf_proof(li_B), b_cols)
    return pm.PlainProof(m, n, k, noise_rank, mp_A, mp_B, None)


def build_proof_bt(A: np.ndarray, Bt: np.ndarray, winning_row: int, winning_col: int,
                   key: bytes, noise_rank: int):
    m, k = A.shape
    n = Bt.shape[0]
    TILE = 16
    a_rows = list(range(winning_row, min(winning_row + TILE, m)))
    b_cols = list(range(winning_col, min(winning_col + TILE, n)))
    tree_A = _merkle_tree(A, key)
    tree_B = _merkle_tree(Bt, key)
    li_A = pm.MerkleTree.compute_leaf_indices_from_rows(a_rows, (m, k))
    li_B = pm.MerkleTree.compute_leaf_indices_from_rows(b_cols, (n, k))
    mp_A = pm.MatrixMerkleProof(tree_A.get_multileaf_proof(li_A), a_rows)
    mp_B = pm.MatrixMerkleProof(tree_B.get_multileaf_proof(li_B), b_cols)
    return pm.PlainProof(m, n, k, noise_rank, mp_A, mp_B, None)


def derive_key(header_bytes: bytes, mining_config) -> bytes:
    return blake3.blake3(header_bytes + mining_config.to_bytes()).digest()


def verify_proof_local(header_bytes: bytes, proof, nbits: int | None = None):
    hdr = pm.IncompleteBlockHeader.from_bytes(header_bytes)
    try:
        if nbits is not None:
            return pm.verify_plain_proof(hdr, proof, nbits)
        return pm.verify_plain_proof(hdr, proof)
    except TypeError:
        return pm.verify_plain_proof(hdr, proof)
