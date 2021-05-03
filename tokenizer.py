import logging

atomic_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
                   'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
                   'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                   'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                   'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                   'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
                   'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']

class MoleculeTokenizer(object):
    def __init__(self) -> None:
        self.elements = set(atomic_elements)
        self.elements.add('OH')
        self.max_len = max(map(len, self.elements))
        self.digits = set(map(str, range(10)))
        super().__init__()

    def tokenize_formula(self, formula):
        i = 0
        tokens = []
        while i < len(formula):
            elem = None
            for j in range(self.max_len, 0, -1):
                x = formula[i:i+j]
                if x in self.elements:
                    elem = x
                    i += j
                    tokens.append(elem)
                    break
            num = None
            j = i
            while j < len(formula):
                if formula[j] in self.digits:
                    j += 1
                else:
                    break
            if j > i:
                num = int(formula[i:j])
                tokens.append(num)
                i = j
            if elem is None and num is None:
                logging.warning('Invalid molecule syntax: ' + formula)
                return []
        return tokens

    @staticmethod
    def to_count_form(tokenized_formula):
      ret = []
      curr_elem = None
      for i in range(len(tokenized_formula)):
          if curr_elem is None:
              curr_elem = tokenized_formula[i]
              if not isinstance(curr_elem, str):
                  logging.warning(
                      'Invalid tokenized formula: ' + str(tokenized_formula))
                  return []
          else:
              num = tokenized_formula[i]
              if isinstance(num, str):
                  ret.append((curr_elem, 1))
                  curr_elem = num
              else:
                  ret.append((curr_elem, num))
                  curr_elem = None
      if not curr_elem is None:
          ret.append((curr_elem, 1))
      return ret
