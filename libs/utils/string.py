import datetime
import fnmatch
import itertools
import random
import re
import typing as T


def delimited_strings(s: str, delimiter: str = ',') -> T.Iterable[str]:
  """Split comma-separated list (ignore whitespace), e.g. '1- 3, a,b ' -> ('1- 3', 'a', 'b')."""
  # TODO: Ignore commas within balanced parentheses/brackets/quotes.
  return (item.strip() for item in s.split(delimiter))


def int_range(spec: str) -> T.Iterable[int]:
  """Expand comma-separated int ranges, e.g. '1-3,5,7-8' -> (1, 2, 3, 5, 7, 8)."""
  for span in delimited_strings(spec):
    bounds = span.split('-')
    if len(bounds) == 2:
      start, end = bounds
      start = int(start)
      end = int(end)
      for value in range(start, end + 1):
        yield value
    elif len(bounds) == 1:
      value = int(bounds[0])
      yield value
    else:
      raise ValueError(f'Invalid int span: {span}')


def char_range(spec: str) -> T.Iterable[str]:
  """Expand comma-separated char ranges, e.g. 'a-c,e' -> ('a', 'b', 'c', 'e')."""
  for span in delimited_strings(spec):
    bounds = span.split('-')
    if len(bounds) == 2:
      start, end = bounds
      start = ord(start)
      end = ord(end)
      for value in range(start, end + 1):
        yield chr(value)
    elif len(bounds) == 1 and len(bounds[0]):
      value = bounds[0]
      yield value
    else:
      raise ValueError(f'Invalid char span: {span}')


def int_char_range(spec: str, stringify=False) -> T.Iterable[T.Union[str, int]]:
  """Expand comma-separated int/char ranges, e.g. 'a-b,1,ab' -> ('a', 'b', 1, 'ab')."""
  # TODO: Enable expansion of zero-padded number, e.g. '08-10' -> ('08', '09', '10').
  for span in delimited_strings(spec):
    if any(c.isdigit() for c in span):
      for value in int_range(span):
        yield str(value) if stringify else value
    elif '-' in span:
      for value in char_range(span):
        yield value
    else:
      yield span


def glob_range(spec: str) -> T.Iterable[str]:
  """Expand glob patterns with [ int/char ranges ] or { string options }, e.g. '{foo,bar}_*_[1-2]'
  -> (foo_*_1, foo_*_2, bar_*_1, bar_*_2).
  """
  # Convert pattern into [constant_1, spec_1, constant_2, spec_2, ..., constant_n].
  pieces = []
  idx = 0
  # Searches for all specs of form '[...]' and '{...}'.
  for expansion in re.finditer(r'[{\[]([^{}\[\]]*)[}\]]', spec):
    sidx, eidx = expansion.span()
    group = expansion.group()
    bracket = group[0]
    inner = group[1:-1]
    constant = spec[idx:sidx]
    pieces.append((constant,))  # Must be tuple for product.
    if bracket == '{':
      pieces.append(delimited_strings(inner))
    elif bracket == '[':
      pieces.append(int_char_range(inner, stringify=True))
    else:
      assert False
    idx = eidx
  constant = spec[idx:]
  pieces.append((constant,))  # Must be tuple for product.
  assert len(pieces) % 2 == 1
  for prod in itertools.product(*pieces):
    yield ''.join(prod)


def glob_ranges(spec: str | T.Iterable[str]) -> T.Iterable[str]:
  if isinstance(spec, str): spec = [spec]
  return itertools.chain.from_iterable(map(glob_range, spec))


def in_glob_range(spec: str | T.Iterable[str], item: str) -> bool:
  """Returns if `test` string matches `spec`, e.g. 'b_foof_1' matches '[a-c]_foo*_[1-3]'."""
  return any(fnmatch.fnmatchcase(item, pattern) for pattern in glob_ranges(spec))


def filter_by_glob_range(spec: str | T.Iterable[str], items: T.Iterable[str]) -> T.Iterable[str]:
  """Returns all `item` strings that matche `spec`, e.g. 'b_foof_1' matches '[a-c]_foo*_[1-3]'."""
  patterns = list(glob_ranges(spec))
  return (item for item in items if any(fnmatch.fnmatchcase(item, pattern) for pattern in patterns))


def timestamp(format: str = '%Y-%m%d-%H%M%S', tz: datetime.tzinfo | None = None) -> str:
  now = datetime.datetime.now(tz=tz)
  return now.strftime(format)


# yapf: disable
ADJECTIVES = (
  'adorable', 'adventurous', 'alluring', 'amazing', 'ambitious', 'amusing', 'astonishing',
  'attractive', 'awesome', 'bashful', 'bawdy', 'beautiful', 'bewildered', 'bizarre', 'bouncy',
  'brainy', 'brave', 'brawny', 'burly', 'capricious', 'careful', 'caring', 'cautious', 'charming',
  'cheerful', 'chivalrous', 'classy', 'clever', 'clumsy', 'colossal', 'cool', 'coordinated',
  'courageous', 'cuddly', 'curious', 'cute', 'daffy', 'dapper', 'dashing', 'dazzling', 'delicate',
  'delightful', 'determined', 'eager', 'embarrassed', 'enchanted', 'energetic', 'enormous',
  'entertaining', 'enthralling', 'enthusiastic', 'evanescent', 'excited', 'exotic', 'exuberant',
  'exultant', 'fabulous', 'fancy', 'festive', 'finicky', 'flashy', 'flippant', 'fluffy',
  'fluttering', 'funny', 'furry', 'fuzzy', 'gaudy', 'gentle', 'giddy', 'glamorous', 'gleaming',
  'goofy', 'gorgeous', 'graceful', 'grandiose', 'groovy', 'handsome', 'happy', 'hilarious',
  'honorable', 'hulking', 'humorous', 'industrious', 'incredible', 'intelligent', 'jazzy', 'jolly',
  'joyous', 'kind', 'macho', 'magnificent', 'majestic', 'marvelous', 'mighty', 'mysterious',
  'naughty', 'nimble', 'nutty', 'oafish', 'obnoxious', 'outrageous', 'pretty', 'psychedelic',
  'psychotic', 'puzzled', 'quirky', 'quizzical', 'rambunctious', 'remarkable', 'sassy', 'shaggy',
  'smelly', 'sneaky', 'spiffy', 'swanky', 'sweet', 'swift', 'talented', 'thundering', 'unkempt',
  'upbeat', 'uppity', 'wacky', 'waggish', 'whimsical', 'wiggly', 'zany',
)

NOUNS = (
  'aardvarks', 'alligators', 'alpacas', 'anteaters', 'antelopes', 'armadillos', 'baboons',
  'badgers', 'bears', 'beavers', 'boars', 'buffalos', 'bulls', 'bunnies', 'camels', 'cats',
  'chameleons', 'cheetahs', 'centaurs', 'chickens', 'chimpanzees', 'chinchillas', 'chipmunks',
  'cougars', 'cows', 'coyotes', 'cranes', 'crickets', 'crocodiles', 'deer', 'dinosaurs', 'dingoes',
  'dogs', 'donkeys', 'dragons', 'elephants', 'elves', 'ferrets', 'flamingos', 'foxes', 'frogs',
  'gazelles', 'giraffes', 'gnomes', 'gnus', 'goats', 'gophers', 'gorillas', 'hamsters',
  'hedgehogs', 'hippopotamuses', 'hobbits', 'hogs', 'horses', 'hyenas', 'ibexes', 'iguanas',
  'impalas', 'jackals', 'jackalopes', 'jaguars', 'kangaroos', 'kittens', 'koalas', 'lambs',
  'lemmings', 'leopards', 'lions', 'ligers', 'lizards', 'llamas', 'lynxes', 'meerkats', 'moles',
  'mongooses', 'monkeys', 'moose', 'mules', 'newts', 'okapis', 'orangutans', 'ostriches', 'otters',
  'oxen', 'pandas', 'panthers', 'peacocks', 'pegasi', 'phoenixes', 'pigeons', 'pigs',
  'platypuses', 'ponies', 'porcupines', 'porpoises', 'pumas', 'pythons', 'rabbits', 'raccoons',
  'rams', 'reindeer', 'rhinoceroses', 'salamanders', 'seals', 'sheep', 'skunks', 'sloths',
  'slugs', 'snails', 'snakes', 'sphinxes', 'sprites', 'squirrels', 'takins', 'tigers', 'toads',
  'trolls', 'turtles', 'unicorns', 'walruses', 'warthogs', 'weasels', 'wolves', 'wolverines',
  'wombats', 'woodchucks', 'yaks', 'zebras',
)

VERBS = (
  'ambled', 'assembled', 'burst', 'babbled', 'charged', 'chewed', 'clamored', 'coasted', 'crawled',
  'crept', 'danced', 'dashed', 'drove', 'flopped', 'galloped', 'gathered', 'glided', 'hobbled',
  'hopped', 'hurried', 'hustled', 'jogged', 'juggled', 'jumped', 'laughed', 'marched', 'meandered',
  'munched', 'passed', 'plodded', 'pranced', 'ran', 'raced', 'rushed', 'sailed', 'sang',
  'sauntered', 'scampered', 'scurried', 'skipped', 'slogged', 'slurped', 'spied', 'sprinted',
  'spurted', 'squiggled', 'squirmed', 'stretched', 'strode', 'strut', 'swam', 'swung', 'traveled',
  'trudged', 'tumbled', 'twisted', 'wade', 'wandered', 'whistled', 'wiggled', 'wobbled', 'yawned',
  'zipped', 'zoomed',
)

ADVERBS = (
  'absentmindedly', 'adventurously', 'angrily', 'anxiously', 'awkwardly', 'bashfully',
  'beautifully', 'bleakly', 'blissfully', 'boastfully', 'boldly', 'bravely', 'briskly', 'calmly',
  'carefully', 'cautiously', 'cheerfully', 'cleverly', 'cluelessly', 'clumsily', 'coaxingly',
  'colorfully', 'coolly', 'courageously', 'curiously', 'daintily', 'defiantly', 'deliberately',
  'delightfully', 'diligently', 'dreamily', 'drudgingly', 'eagerly', 'effortlessly', 'elegantly',
  'energetically', 'enthusiastically', 'excitedly', 'fervently', 'foolishly', 'furiously',
  'gallantly', 'gently', 'gladly', 'gleefully', 'gracefully', 'gratefully', 'happily', 'hastily',
  'haphazardly', 'hungrily', 'innocently', 'inquisitively', 'intensely', 'jokingly', 'jestingly',
  'joyously', 'jovially', 'jubilantly', 'kiddingly', 'knavishly', 'knottily', 'kookily', 'lazily',
  'loftily', 'longingly', 'lovingly', 'loudly', 'loyally', 'madly', 'majestically', 'merrily',
  'mockingly', 'mysteriously', 'nervously', 'noisily', 'obnoxiously', 'oddly', 'optimistically',
  'overconfidently', 'outside', 'owlishly', 'patiently', 'playfully', 'politely', 'powerfully',
  'purposefully', 'quaintly', 'quarrelsomely', 'queasily', 'quickly', 'quietly', 'quirkily',
  'quizzically', 'rapidly', 'reassuringly', 'recklessly', 'reluctantly', 'reproachfully', 'sadly',
  'scarily', 'seriously', 'shakily', 'sheepishly', 'shyly', 'silently', 'sillily', 'sleepily',
  'slowly', 'speedily', 'stealthily', 'sternly', 'suspiciously', 'sweetly', 'tenderly', 'tensely',
  'thoughtfully', 'triumphantly', 'unabashedly', 'unaccountably', 'urgently', 'vainly',
  'valiantly', 'victoriously', 'warmly', 'wearily', 'youthfully', 'zestfully',
)
# yapf: enable


def hruid(n: int = 2, sep: str = '-'):
  assert 1 <= n <= 5
  parts = []
  if n == 5: parts.append(str(random.randint(2, 33)))
  if n >= 1: parts.append(random.choice(ADJECTIVES))
  if n >= 2: parts.append(random.choice(NOUNS))
  if n >= 3: parts.append(random.choice(VERBS))
  if n >= 4: parts.append(random.choice(ADVERBS))
  return sep.join(parts)
