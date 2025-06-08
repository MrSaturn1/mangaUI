#!/usr/bin/env python3
"""
Test script to compare Magi-only vs Hybrid embeddings for character consistency.
"""

import torch
import numpy as np
from pathlib import Path
import json

def load_embeddings_from_map(embeddings_map_path):
    """Load all embeddings from the embeddings map"""
    if not Path(embeddings_map_path).exists():
        print(f"Embeddings map not found: {embeddings_map_path}")
        return {}
    
    with open(embeddings_map_path, 'r') as f:
        embeddings_map = json.load(f)
    
    embeddings = {}
    for character_name, char_data in embeddings_map.items():
        embedding_path = char_data["embedding_path"]
        if Path(embedding_path).exists():
            try:
                embedding = torch.load(embedding_path, map_location='cpu')
                embeddings[character_name] = {
                    'embedding': embedding,
                    'type': char_data.get('embedding_type', 'unknown'),
                    'shape': embedding.shape
                }
            except Exception as e:
                print(f"Error loading embedding for {character_name}: {e}")
    
    return embeddings

def calculate_similarity_matrix(embeddings):
    """Calculate pairwise similarities between all character embeddings"""
    character_names = list(embeddings.keys())
    n_chars = len(character_names)
    
    similarity_matrix = np.zeros((n_chars, n_chars))
    
    for i, char1 in enumerate(character_names):
        for j, char2 in enumerate(character_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                emb1 = embeddings[char1]['embedding']
                emb2 = embeddings[char2]['embedding']
                
                # Flatten if needed
                if emb1.dim() > 1:
                    emb1 = emb1.flatten()
                if emb2.dim() > 1:
                    emb2 = emb2.flatten()
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(emb1, emb2, dim=0)
                similarity_matrix[i, j] = similarity.item()
    
    return similarity_matrix, character_names

def analyze_embedding_quality(embeddings, embedding_type):
    """Analyze the quality of embeddings"""
    print(f"\n=== Analysis for {embedding_type} embeddings ===")
    
    if not embeddings:
        print("No embeddings found")
        return
    
    # Filter embeddings by type
    filtered_embeddings = {name: data for name, data in embeddings.items() 
                          if data['type'] == embedding_type}
    
    if not filtered_embeddings:
        print(f"No {embedding_type} embeddings found")
        return
    
    print(f"Found {len(filtered_embeddings)} {embedding_type} embeddings")
    
    # Calculate similarity matrix
    similarity_matrix, character_names = calculate_similarity_matrix(filtered_embeddings)
    
    # Extract upper triangle (excluding diagonal)
    n = len(character_names)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(similarity_matrix[i, j])
    
    similarities = np.array(similarities)
    
    print(f"Pairwise similarity statistics:")
    print(f"  Mean: {similarities.mean():.4f}")
    print(f"  Std:  {similarities.std():.4f}")
    print(f"  Min:  {similarities.min():.4f}")
    print(f"  Max:  {similarities.max():.4f}")
    
    # Find most and least similar pairs
    max_idx = np.argmax(similarities)
    min_idx = np.argmin(similarities)
    
    # Convert back to character pairs
    pair_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if pair_idx == max_idx:
                print(f"  Most similar pair: {character_names[i]} & {character_names[j]} ({similarities[max_idx]:.4f})")
            if pair_idx == min_idx:
                print(f"  Least similar pair: {character_names[i]} & {character_names[j]} ({similarities[min_idx]:.4f})")
            pair_idx += 1
    
    # Check for potential issues
    if similarities.std() < 0.1:
        print("  ⚠️  Low variance - embeddings might be too similar")
    elif similarities.std() > 0.5:
        print("  ⚠️  High variance - embeddings might be inconsistent")
    else:
        print("  ✅ Good variance in similarity scores")
    
    return similarities

def compare_embedding_approaches():
    """Compare different embedding approaches"""
    embeddings_map_path = "character_output/character_embeddings/character_embeddings_map.json"
    embeddings = load_embeddings_from_map(embeddings_map_path)
    
    if not embeddings:
        print("No embeddings found. Please generate some embeddings first.")
        return
    
    print(f"Loaded {len(embeddings)} character embeddings")
    
    # Analyze different embedding types
    embedding_types = set(data['type'] for data in embeddings.values())
    print(f"Found embedding types: {embedding_types}")
    
    results = {}
    for embedding_type in embedding_types:
        similarities = analyze_embedding_quality(embeddings, embedding_type)
        if similarities is not None:
            results[embedding_type] = similarities
    
    # Compare results
    if len(results) > 1:
        print(f"\n=== Comparison ===")
        for embedding_type, similarities in results.items():
            print(f"{embedding_type}: mean={similarities.mean():.4f}, std={similarities.std():.4f}")
        
        # Simple ranking (lower std deviation is generally better for character consistency)
        ranking = sorted(results.items(), key=lambda x: x[1].std())
        print(f"\nRanking by consistency (lower std = more consistent):")
        for i, (embedding_type, similarities) in enumerate(ranking):
            print(f"  {i+1}. {embedding_type}: std={similarities.std():.4f}")

def test_individual_character(character_name):
    """Test embeddings for a specific character"""
    embeddings_map_path = "character_output/character_embeddings/character_embeddings_map.json"
    embeddings = load_embeddings_from_map(embeddings_map_path)
    
    if character_name not in embeddings:
        print(f"Character '{character_name}' not found in embeddings")
        print(f"Available characters: {list(embeddings.keys())}")
        return
    
    char_data = embeddings[character_name]
    print(f"\n=== Character: {character_name} ===")
    print(f"Embedding type: {char_data['type']}")
    print(f"Embedding shape: {char_data['shape']}")
    
    embedding = char_data['embedding']
    if embedding.dim() > 1:
        embedding = embedding.flatten()
    
    print(f"Embedding stats:")
    print(f"  Mean: {embedding.mean():.4f}")
    print(f"  Std:  {embedding.std():.4f}")
    print(f"  Min:  {embedding.min():.4f}")
    print(f"  Max:  {embedding.max():.4f}")
    print(f"  Norm: {embedding.norm():.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and compare character embeddings")
    parser.add_argument("--compare", action="store_true", help="Compare all embedding approaches")
    parser.add_argument("--character", type=str, help="Test specific character")
    
    args = parser.parse_args()
    
    if args.character:
        test_individual_character(args.character)
    elif args.compare:
        compare_embedding_approaches()
    else:
        print("Use --compare to compare all approaches or --character NAME to test a specific character")