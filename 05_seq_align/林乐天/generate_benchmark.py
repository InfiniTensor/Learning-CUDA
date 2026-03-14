import random
import os

NUM_CHROMOSOMES = 100
CHROM_LENGTH = 100000000
NUM_READS = 10000
READ_LENGTH = 100
MUTATION_RATE = 0.05
UNKNOWN_RATE = 0.1

BASES = ['A', 'C', 'G', 'T']
QUAL_HIGH = 'I'
QUAL_LOW = '!'

def generate_fasta(filename):
    chromosomes = {}
    with open(filename, 'w') as f:
        for i in range(1, NUM_CHROMOSOMES + 1):
            chrom_name = f"chr{i}"
            f.write(f">{chrom_name}\n")
            
            seq = []
            chunk_size = 10000
            for _ in range(0, CHROM_LENGTH, chunk_size):
                chunk = ''.join(random.choices(BASES, k=min(chunk_size, CHROM_LENGTH - len(seq))))
                f.write(chunk + "\n")
                seq.append(chunk)
                
            chromosomes[chrom_name] = ''.join(seq)
    return chromosomes

def mutate_sequence(seq):
    mutated = list(seq)
    quals = [QUAL_HIGH] * len(seq)
    
    # Track the actual start shift if deletions happen at the very beginning
    actual_start_offset = 0 
    
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            op = random.choice(['sub', 'ins', 'del'])
            if op == 'sub':
                mutated[i] = random.choice([b for b in BASES if b != mutated[i]])
                quals[i] = QUAL_LOW
            elif op == 'ins':
                mutated[i] = mutated[i] + random.choice(BASES)
                quals[i] = QUAL_LOW
            elif op == 'del':
                mutated[i] = ''
                quals[i] = ''
                if i == 0: 
                    actual_start_offset += 1 # If first base is deleted, the real start shifts
                
    return ''.join(mutated)[:READ_LENGTH], ''.join(quals)[:READ_LENGTH], actual_start_offset

def generate_fastq_and_truth(chromosomes, fastq_file, truth_file):
    chrom_names = list(chromosomes.keys())
    
    with open(fastq_file, 'w') as fq, open(truth_file, 'w') as tr:
        for i in range(1, NUM_READS + 1):
            read_name = f"read{i}"
            
            if random.random() < UNKNOWN_RATE:
                seq = ''.join(random.choices(BASES, k=READ_LENGTH))
                qual = QUAL_HIGH * READ_LENGTH
                tr.write(f"{read_name} unknown_origin\n")
            else:
                chrom = random.choice(chrom_names)
                start_pos = random.randint(0, CHROM_LENGTH - READ_LENGTH)
                origin_seq = chromosomes[chrom][start_pos:start_pos + READ_LENGTH + 10] # Pad for deletions
                
                seq, qual, offset = mutate_sequence(origin_seq)
                
                while len(seq) < READ_LENGTH:
                    seq += random.choice(BASES)
                    qual += QUAL_HIGH
                    
                final_pos = start_pos + offset
                tr.write(f"{read_name} {chrom} {final_pos}\n")
                
            fq.write(f"@{read_name}\n{seq}\n+\n{qual}\n")

if __name__ == "__main__":
    print("Generating reference genome...")
    chroms = generate_fasta("reference.fasta")
    
    print("Generating reads and ground truth...")
    generate_fastq_and_truth(chroms, "reads.fastq", "truth.txt")
    
    print("Benchmark data generated successfully.")
