#!/usr/bin/env python

# Imports
# -------
from tinydb import Query

# Rdkit Imports
# -------------
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from IPython.display import SVG

# Plotly Imports
# --------------
import plotly.offline as pyo
import plotly.graph_objs as go
import chart_studio.plotly as py

class Compass(object):

    __version__ = '0.0.2'

    '''
    
    Main component class process data and visualize the resulting plot. 
    
    '''

    def __init__(self, databases, smarts,
                 charge_penalty_boundaries = [],
                 atom_type_penalty_boundaries = [],
                 raw_layers=False,
                 just_functional_groups=False,
                 verbose=False):

        """

        Arguments:
            databases (List): list of TinyDB objects used to filter the data
            smarts (Dict): A dict object comprised of IUPAC as key and the corresponding SMARTS strings as the value.
            raw_layers (Bool): Raw layers will return the new layers of the graph.
            just_functional_groups (Bool): Just the functional group outer layer.
            verbose (Bool): If the user would like back more information.

        """


        # Configurations and Data Mine Setup
        # ----------------------------------
        self.databases = databases
        self.verbose = verbose
        self.user = Query()
        self.smarts = self._validate_smarts(smarts)
        self.compounds = self._fetch_compounds()
        self.charge_penalty_boundaries = charge_penalty_boundaries
        self.atom_type_penalty_boundaries = atom_type_penalty_boundaries

        if just_functional_groups:
            print ('Not Implemented Error..')

        # Data Mining
        # -----------
        self.first_layer_labels, self.first_layer_parents = self._first_layer_data_mine()
        self.first_layer_values, self.second_layer_labels, self.second_layer_parents, self.second_layer_values = self._second_layer_data_mine()
        self.third_layer_labels, self.third_layer_parents, self.third_layer_values, self.third_layer_colors = self._third_layer_data_mine()
        self.fourth_layer_labels, self.fourth_layer_parents, self.fourth_layer_values = self._fourth_layer_data_mine()

        # Prepare Data For Compass Graph
        # ------------------------------
        self.labels = self.first_layer_labels + self.second_layer_labels + self.third_layer_labels + self.fourth_layer_labels
        self.parents = self.first_layer_parents + self.second_layer_parents + self.third_layer_parents + self.fourth_layer_parents
        self.values = self.first_layer_values + self.second_layer_values + self.third_layer_values + self.fourth_layer_values
        self.colors = len(self.first_layer_labels) * [''] + len(self.second_layer_labels) * [''] + self.third_layer_colors + len(self.fourth_layer_parents) * ['']

        if raw_layers:
            self.raw_layers = [self.labels, self.parents, self.values, self.colors]

    def _validate_smarts(self, smarts):

        """
        Using RDKit we can validate the SMARTS string.
        Arguments:
            smarts (Dict): A dict object comprised of IUPAC as key and the corresponding SMARTS strings as the value.
        """

        validated_smarts = []
        for key, value in smarts.items():

            try:
                if Chem.MolFromSmarts(value) is not None:
                    validated_smarts.append([key, value, Chem.MolFromSmarts(value)])
            except:
                if self.verbose:
                    print ("Compass SMARTS Parsing Error: Name: %s , Value: %s" % (key, value))
                pass

        if self.verbose:
            print ("Step 1: SMARTS values validated")

        return validated_smarts

    def _fetch_compounds(self):

        """
        Will return the compounds queried in the database. Verified that the SMILES needs to exist before moving on.
        This might have occured at the pre-processing step.
        Arguments:
            self (Object): class object Compass
        """

        compounds = []

        for database in self.databases:

            compounds.append(database.search(self.user.smiles.exists()))

        if self.verbose:
            print ("Step 2: Compounds retrieved from database")

        return sum(compounds, [])

    def _first_layer_data_mine(self):

        """
        Mine the First Layer - Used to map out what functional groups exist within a SMILES structure
        Returns:
            category_first_layer (List): List of labels for the first layer
            category_counts (Dict): Functional Group patterns and their counts
        """

        category_first_layer = []

        for compound in self.compounds:

            # SMILES are not RDKit friendly from enamine database - need to change into canonical version


            smiles = compound['smiles']
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
            molecule = Chem.MolFromSmiles(smiles)

            matches = []

            for i in range(0, len(self.smarts)):

                row = self.smarts[i]
                pattern_name = row[0]
                pattern_mol = row[2]

                try:
                    substructs = molecule.GetSubstructMatches(pattern_mol)
                    if substructs:
                        matches.append(pattern_name)

                        if pattern_name not in category_first_layer:
                            category_first_layer.append(pattern_name)
                except:
                    pass

            compound['matches'] = matches

        category_first_layer_labels = ['Kernel 1'] + category_first_layer
        category_first_layer_parents = [""] + ["Kernel 1"] * len(category_first_layer)

        if self.verbose:
            print ("Step 3: First Layer Data Mine Completed")

        return category_first_layer_labels, category_first_layer_parents

    def _second_layer_data_mine(self):


        """
        Relations of Functional Groups with each other.
        Returns:
            total_values (Dict): Total value count of all the matches and substructure matching.
            second_labels (List): List of the category second layer labels.
            second_parents (List): First layer parents that the second layer connects into.
            second_values (List): Second Layer values that exist.
        """

        submatches = {}

        for compound in self.compounds:

            matches = compound['matches']

            for match in matches:
                relations = [match + ' - ' + i for i in matches if i != match]

                for relation in relations:
                    if relation not in submatches:
                        submatches[relation] = 1
                    else:
                        submatches[relation] += 1

                compound['relations'] = relations

        second_labels = []
        second_parents = []
        second_values = []

        for key, value in submatches.items():

            parent = key.split('-')[0].strip()
            label = key

            second_parents.append(parent)
            second_labels.append(label)
            second_values.append(value)

        total_values = {}

        for i in range(0, len(second_labels)):

            parent = second_labels[i].split('-')[0].strip()

            if parent not in total_values:
                total_values[parent] = 0

            total_values[parent] += second_values[i]

        if self.verbose:
            print ("Step 4: Second Layer Data Mine Completed")

        first_layer_values = [sum(total_values.values())] + list(total_values.values())

        return first_layer_values, second_labels, second_parents, second_values

    def _third_layer_data_mine(self):

        """
        Third Layer Data Mine distributing the scores into No, Low, Mid, High
        Returns:
            third_layer_labels (List): List of the category second layer labels.
            third_layer_parents (List): First layer parents that the second layer connects into.
            third_layer_values (List): Second Layer values that exist.
        """

        total_penalties = []

        for compound in self.compounds:

            atom_penalties = compound['charge_atom_penalties']
            total_penalty = 0
            max_atom_type_penalty_score = 0

            for atom_penalty in atom_penalties:

                if atom_penalty[2] != '!':

                    if float(atom_penalty[2]) > max_atom_type_penalty_score:


                        total_penalty += float(atom_penalty[2])

                total_penalties.append(total_penalty)
                compound['total_charge_penalty'] = total_penalty

        charge_scores = {}

        for compound in self.compounds:
            matches = compound['matches']

            for match in matches:
                relations = [match + ' - ' + i for i in matches if i != match]

                for relation in relations:

                    if relation not in charge_scores:
                        charge_scores[relation] = [0, 0, 0, 0]

                    if float(compound['total_charge_penalty']) == self.charge_penalty_boundaries[0]:
                        charge_scores[relation][0] += 1
                    elif self.charge_penalty_boundaries[0] < float(compound['total_charge_penalty']) <= self.charge_penalty_boundaries[1]:
                        charge_scores[relation][1] += 1
                    elif self.charge_penalty_boundaries[1] < float(compound['total_charge_penalty']) <= self.charge_penalty_boundaries[2]:
                        charge_scores[relation][2] += 1
                    elif self.charge_penalty_boundaries[2] < float(compound['total_charge_penalty']):
                        charge_scores[relation][3] += 1

        third_layer_labels = sum([['No - ' + i, 'Low - ' + i, 'Mid - ' + i, 'High - ' + i] for i in self.second_layer_labels], [])
        third_layer_parents = sum([[i] * 4 for i in self.second_layer_labels], [])
        third_layer_values = sum([charge_scores[i] for i in self.second_layer_labels], [])
        third_layer_colors = []

        for i in third_layer_labels:
            if 'No' in i or 'Low' in i or 'Mid' in i:
                third_layer_colors.append('')
            if 'High' in i:
                third_layer_colors.append('#ff5f00')

        if self.verbose:
            print ("Step 5: Third Layer Data Mine Completed")

        return third_layer_labels, third_layer_parents, third_layer_values, third_layer_colors


    def _fourth_layer_data_mine(self):

        """
        Fourth Layer Data Mine
        Returns:
            fourth_layer_labels (List): List of labels for the fourth category
            fourth_layer_parents (List): List of parents from the third layer
            fourth_layer_values (List): Atom type values that exist within this set.
        """

        atom_types = {}

        for compound in self.compounds:

            if 'relations' in compound:
                for relation in compound['relations']:
                    for atom_type in compound['charge_atom_penalties']:

                        # Skip Errors - TODO: verify LP1 and LPH - halogens?
                        if atom_type[2] == '!':
                            continue

                        atom_type_name = atom_type[0]
                        atom_type_score = float(atom_type[2])

                        if  atom_type_score == self.atom_type_penalty_boundaries[0]:
                            charge_score_result = 'No'
                        elif self.atom_type_penalty_boundaries[0] <  atom_type_score <= self.atom_type_penalty_boundaries[1]:
                            charge_score_result = 'Low'
                        elif self.atom_type_penalty_boundaries[1] <  atom_type_score <= self.atom_type_penalty_boundaries[2]:
                            charge_score_result = 'Mid'
                        elif self.atom_type_penalty_boundaries[2] < atom_type_score:
                            charge_score_result = 'High'

                        key = charge_score_result + ' - ' + relation.split(' - ')[1] + ' - ' + relation.split(' - ')[0] + ' - ' + atom_type_name

                        if key not in atom_types:
                            atom_types[key] = 0
                        atom_types[key] += 1

        fourth_layer_labels = []
        fourth_layer_parents = []
        fourth_layer_values = []

        for key, value in atom_types.items():

            fourth_layer_parents.append(' - '.join(key.split(' - ')[0:3]))
            fourth_layer_labels.append(key)
            fourth_layer_values.append(value)

        if self.verbose:
            print ("Step 6: Fourth Layer Data Mine Completed")

        return fourth_layer_labels, fourth_layer_parents, fourth_layer_values

    def visualize(self, renderer=''):

        """
        Generate the CgenFF Compass

        """

        fig =go.Figure()

        fig.add_trace(go.Sunburst(
            labels=self.labels,
            parents=self.parents,
            values=self.values,
            insidetextorientation='radial',
            marker={"colors": self.colors},
            domain=dict(column=0),
            opacity=0.9
        ))

        fig.update_layout(margin = dict(t=5, l=5, r=5, b=5), height=900, width=900,
                          grid= dict(columns=1, rows=1))

        if len(renderer) > 0:
            fig.show(renderer=renderer)

        fig.show()

    @staticmethod
    def moltosvg(mol,molSize=(450,150),kekulize=True):

        mc = Chem.Mol(mol.ToBinary())

        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)

        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        return svg

    @staticmethod
    def render_svg(svg):

        return SVG(svg.replace('svg:',''))

    @staticmethod
    def visualize_smiles(smiles):

        '''
        Produce an image of resultant query
        Arguments:
            query_result (JSON): data collected back from the search query
        '''

        molecules = [Chem.MolFromSmiles(i) for i in smiles]
        for mol in molecules:
            Compass.render_svg(Compass.moltosvg(mol))

        img = Draw.MolsToGridImage(molecules, molsPerRow=4, subImgSize=(250, 250), maxMols=len(smiles),legends=['Compound %s' % i for i in range(1, len(molecules) + 1)], useSVG=False)

        return img

    def query(self, relationship, score_category, render=False, display_atom_type_scores=False, return_table=False, **kwargs):

        """

        This function will query the graph after the data analysis occurs

        Arguments:
            relationship (String): Relationship you are interested in.
            score_category (String): Could be No, Low, Mid, or High.
            render (Bool): Whether to render the graph.
            display_atom_type_scores (Bool): Whether to display the atom type scores from BeautifulTable
            return_table (Bool): Whether to return the table in BeautifulTable Format
        Returns:
            matched_smiles (List or SVG): returns a list of matched smiles for the relationship.
            visualizer (Compounds Represented in IPython): Visualizer off RDKit for visualizing the compounds. Rendered into SVG

        """

        matched_smiles = []

        for compound in self.compounds:

            score = float(compound['bonded_penalty'])

            if 'relations' in compound:
                if relationship in compound['relations']:
                    if score_category == 'high' and self.charge_penalty_boundaries[2] < score:
                        matched_smiles.append(compound['smiles'])
                    elif score_category == 'mid' and self.charge_penalty_boundaries[1] < score < self.charge_penalty_boundaries[2]:
                        matched_smiles.append(compound['smiles'])
                    elif score_category == 'low' and 0 < score < self.charge_penalty_boundaries[1]:
                        matched_smiles.append(compound['smiles'])
                    elif score_category == 'no' and score == 0:
                        matched_smiles.append(compound['smiles'])



                        # Additional Filters

        if 'filters' in kwargs:

            if 'number_of_heavy_atoms' in kwargs['filters']:

                heavy_atom_filtered_smiles = []

                for smiles in matched_smiles:
                    molecule = Chem.MolFromSmiles(smiles)
                    if molecule.GetNumHeavyAtoms() <= kwargs['filters']['number_of_heavy_atoms']:
                        heavy_atom_filtered_smiles.append(smiles)

            if 'atom_types' in kwargs['filters']:

                atom_type_filtered_smiles = []
                atom_type_legends = []

                for compound in self.compounds:

                    if compound['smiles'] in heavy_atom_filtered_smiles:
                        atom_types = [i[0] for i in compound['charge_atom_penalties']]

                        check =  all(item in atom_types for item in kwargs['filters']['atom_types'])

                        if check:
                            atom_type_filtered_smiles.append(compound['smiles'])
                            atom_type_legends.append(compound['charge_atom_penalties'])




        matched_smiles = atom_type_filtered_smiles

        from beautifultable import BeautifulTable

        table = BeautifulTable()
        table.column_headers = ["Compound #", "Atom Type", "Charge", "Penalty"]

        for i in range(0, len(atom_type_legends)):

            for atom_type in atom_type_legends[i]:

                if atom_type[0][0] == 'H':
                    continue

                table.append_row(['Compound %s' % (i + 1)] + atom_type)


        if display_atom_type_scores:
            print (table)

        if render:
            return Compass.visualize_smiles(matched_smiles)
        elif return_table:
            return table
        else:
            return matched_smiles

