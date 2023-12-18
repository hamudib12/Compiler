#use "pc.ml";;

exception X_not_yet_implemented;;
exception X_this_should_not_happen of string;;

let rec is_member a = function
  | [] -> false
  | a' :: s -> (a = a') || (is_member a s);;

let rec gcd a b =
  match (a, b) with
  | (0, b) -> b
  | (a, 0) -> a
  | (a, b) -> gcd b (a mod b);;

type scm_number =
  | ScmRational of (int * int)
  | ScmReal of float;;

type sexpr =
  | ScmVoid
  | ScmNil
  | ScmBoolean of bool
  | ScmChar of char
  | ScmString of string
  | ScmSymbol of string
  | ScmNumber of scm_number
  | ScmVector of (sexpr list)
  | ScmPair of (sexpr * sexpr);;

module type READER = sig
  val nt_sexpr : sexpr PC.parser
  val print_sexpr : out_channel -> sexpr -> unit
  val print_sexprs : out_channel -> sexpr list -> unit
  val sprint_sexpr : 'a -> sexpr -> string
  val sprint_sexprs : 'a -> sexpr list -> string
  val scheme_sexpr_list_of_sexpr_list : sexpr list -> sexpr
end;; (* end of READER signature *)

module Reader : READER = struct
  open PC;;

  type string_part =
    | Static of string
    | Dynamic of sexpr;;

  (*unitfy: it takes a parser, what ever the parser.found will be, it makes it empty tuple*)
  let unitify nt = pack nt (fun _ -> ());;

  (*nt_whitespace: parser that detect whitespaces*)
  let rec nt_whitespace str =
    const (fun ch -> ch <= ' ') str

  (*nt_end_of_line_or_file: parser that detect the end of the line*)
  and nt_end_of_line_or_file str =
    let null_terminator = unitify (char '\n') in
    let end_of_input = unitify nt_end_of_input in
    let nt = disj null_terminator end_of_input in
    nt str

  and nt_line_comment str =
    let semicolon = char ';' in
    let not_end_of_file = diff nt_any nt_end_of_line_or_file in
    let not_end_of_file = star not_end_of_file in
    let nt = caten semicolon not_end_of_file in
    let nt = caten nt nt_end_of_line_or_file in
    let nt = unitify nt in
    nt str

  and every_thing_except_braces str =
    let nt1 = disj_list [unitify nt_char ; unitify nt_string ; nt_comment] in
    let nt2 = disj nt1 (unitify (one_of "{}")) in
    let nt2 = diff nt_any nt2 in 
    let nt1 = star (disj (unitify nt2) nt1) in
    nt1 str

    (*comments between {}*)
  and nt_paired_comment str = 
    let nt_left = char '{' in
    let nt_everything_else = every_thing_except_braces in
    let nt_right = char '}' in
    let nt = unitify (caten nt_left (caten nt_everything_else nt_right)) in
    nt str

    (*comments that stops after s-expression *)
  and nt_sexpr_comment str = 
    let nt = unitify (caten (word "#;") nt_sexpr) in
    nt str

    (*all comments*)
  and nt_comment str =
    disj_list[nt_line_comment;
                nt_paired_comment;
                    nt_sexpr_comment] str

  and nt_void str =
    let nt = word_ci "#void" in
    let nt = not_followed_by nt nt_symbol_char in
    let nt = pack nt (fun _ -> ScmVoid) in
    nt str


  and nt_skip_star str =
    let nt = disj (unitify nt_whitespace) nt_comment in
    let nt = unitify (star nt) in
    nt str


  and make_skipped_star (nt : 'a parser) =
    let nt1 = caten nt_skip_star (caten nt nt_skip_star) in
    let nt1 = pack nt1 (fun (_, (e, _)) -> e) in
    nt1

  and nt_digit str = 
    let nt = pack (range '0' '9') (fun x -> (int_of_char x) - (int_of_char '0')) in
    nt str

  and nt_hex_letters str = 
    let nt = pack (range 'a' 'f') (fun x -> ((int_of_char x) - (int_of_char 'a') + 10)) in
    nt str

  and nt_hex_digit str =
    let nt = disj nt_digit nt_hex_letters in
    let nt = star nt in
    nt str

    (*get natural numbers by getting the (old_int * 10) + new_int*)
  and nt_nat str = 
    let nt = pack (plus nt_digit) (fun list -> List.fold_left
                                              (fun base ele -> 10 * base + ele) 
                                              0
                                              list) in 
    nt str

  and nt_hex_nat str = 
    let nt = pack nt_hex_digit
                (fun digits ->
                  List.fold_left
                    (fun num digit ->
                      16 * num + digit)
                    0
                    digits) in
    nt str

  and nt_optional_sign str = 
    let nt1 = pack (char '-') (fun (l) -> false) in
    let nt2 = pack (char '+') (fun (l) -> true) in
    let nt3 = pack (word "") (fun (l) -> true) in
    let nt = (disj nt1 (disj nt2 nt3)) in
    nt str

  and nt_int str =
    let nt = caten nt_optional_sign nt_nat in
    let nt = pack nt
                (fun (is_positive, n) ->
                  if is_positive then n else -n) in
    nt str

  and nt_frac str =
    let nt_numerator = caten nt_int (char '/') in
    let nt_just_numerator = pack nt_numerator (fun (num, _) -> num) in
    let nt_denominator = only_if nt_nat (fun n -> n != 0) in
    let nt = caten nt_just_numerator nt_denominator in
    let nt = pack nt
                (fun (num, den) ->
                  let d = gcd num den in
                  ScmRational(num / d, den / d)) in
    nt str

  and nt_integer_part str =
    let nt = plus nt_digit in
    let nt = pack nt
                (fun digits ->
                  List.fold_left
                    (fun num digit -> 10.0 *. num +. (float_of_int digit))
                    0.0
                    digits) in
    nt str

  and nt_mantissa str =
    let nt = plus nt_digit in
    let nt = pack nt
                (fun digits ->
                  List.fold_right
                    (fun digit num ->
                      ((float_of_int digit) +. num) /. 10.0)
                    digits
                    0.0) in
    nt str

  and nt_exponent str =
    let nt1 = unitify (char_ci 'e') in
    let nt2 = word "*10" in
    let nt3 = unitify (word "**") in
    let nt4 = unitify (char '^') in
    let nt3 = disj nt3 nt4 in
    let nt2 = caten nt2 nt3 in
    let nt2 = unitify nt2 in
    let nt1 = disj nt1 nt2 in
    let nt1 = caten nt1 nt_int in
    let nt1 = pack nt1 (fun (_, n) -> Float.pow 10. (float_of_int n)) in
    nt1 str


  and make_maybe nt none_value =
    pack (maybe nt)
      (function
       | None -> none_value
       | Some(x) -> x)
  

  and nt_float_A str= 
    let nt1 = nt_integer_part in              
    let nt2 = char '.' in
    let nt3 = caten (make_maybe nt_mantissa 0.0) (make_maybe nt_exponent 1.0) in 
    let nt2 = caten nt2 nt3 in 
    let nt1 = caten nt1 nt2 in 
    let nt1 = pack nt1 (fun (integer_p,(_, (mantisa, expo))) -> (integer_p +. mantisa) *. expo) in 
    nt1 str 
    
  and nt_float_B str=               
    let nt1 = char '.' in
    let nt2 = caten nt_mantissa  (make_maybe nt_exponent 1.0) in 
    let nt1 = caten nt1 nt2 in 
    let nt1 = pack nt1 (fun (_, (mantisa, expo)) -> mantisa *. expo) in 
    nt1 str 
   
  and nt_float_C str=    
    let nt1 = caten nt_integer_part nt_exponent in 
    let nt1 = pack nt1 (fun (integer_p, expo) -> integer_p *. expo) in 
    nt1 str 
   
    

  and nt_float str =
    let nt1 = nt_optional_sign in                  
    let nt2 = (disj nt_float_A (disj nt_float_B nt_float_C)) in 
    let nt1 = pack (caten nt1 nt2) (fun (is_positive , number) -> 
                                    if is_positive then ScmReal(number) else ScmReal (-. number)) in
    nt1 str

  and nt_number str =
    let nt1 = nt_float in
    let nt2 = nt_frac in
    let nt3 = pack nt_int (fun n -> ScmRational(n, 1)) in
    let nt1 = disj nt1 (disj nt2 nt3) in
    let nt1 = pack nt1 (fun r -> ScmNumber r) in
    let nt1 = not_followed_by nt1 nt_symbol_char in
    nt1 str

  and nt_boolean str =
    let nt1 = char '#' in
    let nt2 = char_ci 'f' in
    let nt2 = pack nt2 (fun _ -> ScmBoolean false) in
    let nt3 = char_ci 't' in
    let nt3 = pack nt3 (fun _ -> ScmBoolean true) in
    let nt2 = disj nt2 nt3 in
    let nt1 = caten nt1 nt2 in
    let nt1 = pack nt1 (fun (_, value) -> value) in
    let nt2 = nt_symbol_char in
    let nt1 = not_followed_by nt1 nt2 in
    nt1 str

  and nt_char_simple str =
    let nt1 = const(fun ch -> ' ' < ch) in
    let nt1 = not_followed_by nt1 nt_symbol_char in
    nt1 str

  and nt_char_named str = 
    let nt1 = pack (word_ci "newline") (fun _ -> '\n') in
    let nt2 = pack (word_ci "nul") (fun _ -> '\000') in
    let nt3 = pack (word_ci "page") (fun _ -> '\012') in
    let nt4 = pack (word_ci "return") (fun _ -> '\r') in
    let nt5 = pack (word_ci "space") (fun _ -> ' ') in
    let nt6 = pack (word_ci "tab") (fun _ -> '\t') in
    let nt1 = (disj nt1 (disj nt2 (disj nt3 (disj nt4 (disj nt5 nt6))))) in
    nt1 str

  and nt_char_hex str =
    let nt1 = caten (char_ci 'x') nt_hex_nat in
    let nt1 = pack nt1 (fun (_, n) -> n) in
    let nt1 = only_if nt1 (fun n -> n < 256) in
    let nt1 = pack nt1 (fun n -> char_of_int n) in
    nt1 str

  and nt_char str =
    let nt1 = word "#\\" in
    let nt2 = disj nt_char_simple (disj nt_char_named nt_char_hex) in
    let nt1 = caten nt1 nt2 in
    let nt1 = pack nt1 (fun (_, ch) -> ScmChar ch) in
    nt1 str

  and nt_symbol_char str =
    let nt1 = range_ci 'a' 'z' in
    let nt1 = pack nt1 Char.lowercase_ascii in
    let nt2 = range '0' '9' in
    let nt3 = one_of "!$^*_-+=<>?/" in
    let nt1 = disj nt1 (disj nt2 nt3) in
    nt1 str

  and nt_symbol str = 
    let nt1 = plus nt_symbol_char in
    let nt1 = pack (pack nt1 string_of_list) (fun s -> ScmSymbol s) in
    nt1 str

  and nt_string_part_simple str =
    let nt1 =
      disj_list [unitify (char '"'); unitify (char '\\'); unitify (word "~~");
                 unitify nt_string_part_dynamic] in
    let nt1 = diff nt_any nt1 in
    nt1 str

  and nt_string_part_meta str =
    let nt1 =
      disj_list [pack (word "\\\\") (fun _ -> '\\');
                 pack (word "\\\"") (fun _ -> '"');
                 pack (word "\\n") (fun _ -> '\n');
                 pack (word "\\r") (fun _ -> '\r');
                 pack (word "\\f") (fun _ -> '\012');
                 pack (word "\\t") (fun _ -> '\t');
                 pack (word "~~") (fun _ -> '~')] in
    nt1 str

  and nt_string_part_hex str =
    let nt1 = word_ci "\\x" in
    let nt2 = nt_hex_nat in
    let nt2 = only_if nt2 (fun n -> n < 256) in
    let nt3 = char ';' in
    let nt1 = caten nt1 (caten nt2 nt3) in
    let nt1 = pack nt1 (fun (_, (n, _)) -> n) in
    let nt1 = pack nt1 char_of_int in
    nt1 str

  and nt_string_part_dynamic str = 
    let nt1 = word "~{" in 
    let nt2 = nt_skip_star in
    let nt3 = word "}" in 
    let nt1 = caten (caten nt1 nt2) nt_sexpr in 
    let nt1 = caten nt1 nt2 in 
    let nt1 = caten nt1 nt3 in 
    let nt1 = pack nt1 (fun ((((_ ,_), sexp),_),_) -> (Dynamic (ScmPair(ScmSymbol("format"), ScmPair(ScmString("~a"), ScmPair (sexp, ScmNil)))))) in
    nt1 str

  and nt_string_part_static str =
    let nt1 = disj_list [nt_string_part_simple;
                         nt_string_part_meta;
                         nt_string_part_hex] in
    let nt1 = plus nt1 in
    let nt1 = pack nt1 string_of_list in
    let nt1 = pack nt1 (fun str -> Static str) in
    nt1 str

  and nt_string_part str =
    disj nt_string_part_static nt_string_part_dynamic str

  and nt_string str =
    let nt1 = char '"' in
    let nt2 = star nt_string_part in
    let nt3 = char '"' in
    let nt1 = caten nt1 (caten nt2 nt3) in
    let nt1 = pack nt1 (fun (_, (parts, _)) -> parts) in
    let nt1 = pack nt1
                (fun parts ->
                  match parts with
                  | [] -> ScmString ""
                  | [Static(str)] -> ScmString str
                  | [Dynamic(sexpr)] -> sexpr
                  | parts ->
                     let argl =
                       List.fold_right
                         (fun car cdr ->
                           ScmPair((match car with
                                    | Static(str) -> ScmString(str)
                                    | Dynamic(sexpr) -> sexpr),
                                   cdr))
                         parts
                         ScmNil in
                     ScmPair(ScmSymbol "string-append", argl)) in
    nt1 str

  and nt_vector str = 
    let empty = caten nt_skip_star (char ')') in
    let empty = pack empty (fun _ -> ScmVector []) in
    let nt1 = star nt_sexpr in
    let nt1 = caten nt1 (char ')') in
    let nt1 = pack nt1 (fun (s, _) -> ScmVector s) in
    let nt2 = disj empty nt1 in
    let nt2 = caten (word "#(") nt2 in
    let nt2 = pack nt2 (fun (_, s) -> s) in
    nt2 str                     

  and proper_list str = 
    let empty = pack (caten nt_skip_star (char ')')) (fun _ -> ScmNil) in
    let nt1 = star nt_sexpr in
    let nt1 = caten nt1 (char ')') in
    let nt1 = pack nt1 (fun (s, _) ->  List.fold_right 
                                                  (fun head tail -> ScmPair(head,tail))
                                            s ScmNil) in
    let nt2 = disj empty nt1 in
    let nt2 = pack (caten (char '(') nt2) (fun (_, sexpr) -> sexpr) in
    nt2 str

  and improper_list str =
    let dot = char '.' in
    let nt1 = caten (caten (caten (caten (caten (caten (char '(') (plus nt_sexpr)) nt_skip_star) dot) nt_skip_star) nt_sexpr) (char ')') in
    let nt1 = pack nt1 (fun ((((((_,sexprs),_),dot),_),sexpr), _) ->  List.fold_right (fun hd tl -> ScmPair(hd,tl))
                                                              sexprs sexpr) in
    nt1 str

  and nt_list str = 
    let nt1 = disj proper_list improper_list in
    nt1 str

  and make_quoted_form nt_qf qf_name =
    let nt1 = caten nt_qf nt_sexpr in
    let nt1 = pack nt1
                (fun (_, sexpr) ->
                  ScmPair(ScmSymbol qf_name,
                          ScmPair(sexpr, ScmNil))) in
    nt1

  and nt_quoted_forms str =
    let nt1 =
      disj_list [(make_quoted_form (unitify (char '\'')) "quote");
                 (make_quoted_form (unitify (char '`')) "quasiquote");
                 (make_quoted_form
                    (unitify (not_followed_by (char ',') (char '@')))
                    "unquote");
                 (make_quoted_form (unitify (word ",@")) "unquote-splicing")] in
    nt1 str
  and nt_sexpr str = 
    let nt1 =
      disj_list [nt_void; nt_number; nt_boolean; nt_char; nt_symbol;
                 nt_string; nt_vector; nt_list; nt_quoted_forms] in
    let nt1 = make_skipped_star nt1 in
    nt1 str;;

  let rec string_of_sexpr = function
    | ScmVoid -> "#<void>"
    | ScmNil -> "()"
    | ScmBoolean(false) -> "#f"
    | ScmBoolean(true) -> "#t"
    | ScmChar('\n') -> "#\\newline"
    | ScmChar('\r') -> "#\\return"
    | ScmChar('\012') -> "#\\page"
    | ScmChar('\t') -> "#\\tab"
    | ScmChar(' ') -> "#\\space"
    | ScmChar(ch) ->
       if (ch < ' ')
       then let n = int_of_char ch in
            Printf.sprintf "#\\x%x" n
       else Printf.sprintf "#\\%c" ch
    | ScmString(str) ->
       Printf.sprintf "\"%s\""
         (String.concat ""
            (List.map
               (function
                | '\n' -> "\\n"
                | '\012' -> "\\f"
                | '\r' -> "\\r"
                | '\t' -> "\\t"
                | '\"' -> "\\\""
                | ch ->
                   if (ch < ' ')
                   then Printf.sprintf "\\x%x;" (int_of_char ch)
                   else Printf.sprintf "%c" ch)
               (list_of_string str)))
    | ScmSymbol(sym) -> sym
    | ScmNumber(ScmRational(0, _)) -> "0"
    | ScmNumber(ScmRational(num, 1)) -> Printf.sprintf "%d" num
    | ScmNumber(ScmRational(num, -1)) -> Printf.sprintf "%d" (- num)
    | ScmNumber(ScmRational(num, den)) -> Printf.sprintf "%d/%d" num den
    | ScmNumber(ScmReal(x)) -> Printf.sprintf "%f" x
    | ScmVector(sexprs) ->
       let strings = List.map string_of_sexpr sexprs in
       let inner_string = String.concat " " strings in
       Printf.sprintf "#(%s)" inner_string
    | ScmPair(ScmSymbol "quote",
              ScmPair(sexpr, ScmNil)) ->
       Printf.sprintf "'%s" (string_of_sexpr sexpr)
    | ScmPair(ScmSymbol "quasiquote",
              ScmPair(sexpr, ScmNil)) ->
       Printf.sprintf "`%s" (string_of_sexpr sexpr)
    | ScmPair(ScmSymbol "unquote",
              ScmPair(sexpr, ScmNil)) ->
       Printf.sprintf ",%s" (string_of_sexpr sexpr)
    | ScmPair(ScmSymbol "unquote-splicing",
              ScmPair(sexpr, ScmNil)) ->
       Printf.sprintf ",@%s" (string_of_sexpr sexpr)
    | ScmPair(car, cdr) ->
       string_of_sexpr' (string_of_sexpr car) cdr
  and string_of_sexpr' car_string = function
    | ScmNil -> Printf.sprintf "(%s)" car_string
    | ScmPair(cadr, cddr) ->
       let new_car_string =
         Printf.sprintf "%s %s" car_string (string_of_sexpr cadr) in
       string_of_sexpr' new_car_string cddr
    | cdr ->
       let cdr_string = (string_of_sexpr cdr) in
       Printf.sprintf "(%s . %s)" car_string cdr_string;;

  let print_sexpr chan sexpr = output_string chan (string_of_sexpr sexpr);;

  let print_sexprs chan sexprs =
    output_string chan
      (Printf.sprintf "[%s]"
         (String.concat "; "
            (List.map string_of_sexpr sexprs)));;

  let sprint_sexpr _ sexpr = string_of_sexpr sexpr;;

  let sprint_sexprs chan sexprs =
    Printf.sprintf "[%s]"
      (String.concat "; "
         (List.map string_of_sexpr sexprs));;

  let scheme_sexpr_list_of_sexpr_list sexprs =
    List.fold_right (fun car cdr -> ScmPair (car, cdr)) sexprs ScmNil;;

end;; (* end of struct Reader *)
