// macOS dependency-free JavaScript checks:
// osascript -l JavaScript tests/check_frontend.js "$PWD/static/app.js"
ObjC.import('Foundation');
function run(argv) {
  var source = ObjC.unwrap($.NSString.stringWithContentsOfFileEncodingError(argv[0], $.NSUTF8StringEncoding, null));
  new Function(source); // Parse all production JavaScript without needing a DOM.
  function extract(name) {
    var start = source.indexOf('function '+name+'(');
    var brace = source.indexOf('{', start), depth = 1, end = brace+1;
    while (depth && end < source.length) {
      if (source[end] === '{') depth++;
      if (source[end] === '}') depth--;
      end++;
    }
    return source.slice(start,end);
  }
  var code = extract('validNumber')+'\n'+extract('localToday')+'\n'+extract('goalPlan');
  var test = new Function(code + '\n' + [
    'function assert(ok, message) { if(!ok) throw Error(message); }',
    'assert(validNumber(0) && validNumber(123.45), "valid amounts");',
    'assert(!validNumber(-1) && !validNumber(Infinity) && !validNumber(NaN), "invalid amounts");',
    'var date = new Date(localToday()); date.setDate(date.getDate()+180);',
    'var d = date.getFullYear()+"-"+String(date.getMonth()+1).padStart(2,"0")+"-"+String(date.getDate()).padStart(2,"0");',
    'var p=goalPlan({date:d,target:1200,current:300});',
    'assert(p.months===6 && p.monthly===150 && p.percent===25, "six-month contribution and progress");',
    'var done=goalPlan({date:d,target:100,current:120});',
    'assert(done.monthly===0 && done.percent===100, "completed goal");',
    'var expired=goalPlan({date:"2020-01-01",target:100,current:0});',
    'assert(expired.days<0 && expired.months===1, "expired goal detection");'
  ].join('\n'));
  test();
  return 'PASS: frontend syntax and 5 calculator checks';
}
