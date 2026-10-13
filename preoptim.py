from src.protein import Protein
import argparse


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_PDB_file',
                        type=str,
                        required=True,
                        help='PDB file with structure, which should be preoptimised.')
    parser.add_argument('--output_PDB_file',
                        type=str,
                        required=True,
                        help='PDB file to store preoptimized structure.')
    parser.add_argument('--parameters',
                        type=str,
                        required=True,
                        help='JSON file with PROPTIMUS PREOPTIM parameters.')
    args = parser.parse_args()
    return args


def main():
    args = load_arguments()
    protein = Protein(args.input_PDB_file)
    protein.preoptimize(args.parameters)
    protein.save(args.output_PDB_file)

if __name__ == '__main__':
    main()


# napsat interreziduální interakce
# napsat dihedrální úhly pro vodíky
# napsat volbu pro variabilní vodíky