use nom::IResult;
use std::collections::HashMap;
use std::io::Result;

/// Representation of a Numpy type
#[derive(PartialEq, Eq, Debug)]
pub enum DType {
    /// A simple array with only a single field
    Plain {
        /// Numpy type string. First character is `'>'` for big endian, `'<'` for little endian.
        ///
        /// Examples: `>i4`, `<u8`, `>f8`. The number corresponds to the number of bytes.
        ty: String,

        /// Shape of a type.
        ///
        /// Scalar has zero entries. Otherwise, number of entries == number of dimensions and each
        /// entry specifies size in the respective dimension.
        shape: Vec<u64>,
    },

    /// A structure record array
    Record(Vec<Field>),
}

#[derive(PartialEq, Eq, Debug)]
/// A field of a record dtype
pub struct Field {
    /// The name of the field
    pub name: String,

    /// The dtype of the field
    pub dtype: DType,
}

impl DType {
    /// Numpy format description of record dtype.
    pub fn descr(&self) -> String {
        use crate::DType::*;
        match *self {
            Record(ref fields) => {
                fields
                    .iter()
                    .map(|Field { name, dtype }| match *dtype {
                        Plain { ref ty, ref shape } => {
                            if shape.is_empty() {
                                format!("('{name}', '{ty}'), ")
                            } else {
                                let shape_str = shape
                                    .iter()
                                    .fold(String::new(), |o, n| o + &format!("{n},"));
                                format!("('{name}', '{ty}', ({shape_str})), ")
                            }
                        }
                        ref record @ Record(_) => {
                            format!("('{}', {}), ", name, record.descr())
                        }
                    })
                    .fold("[".to_string(), |o, n| o + &n)
                    + "]"
            }
            Plain { ref ty, .. } => format!("'{ty}'"),
        }
    }

    /// Create from description AST
    pub fn from_descr(descr: Value) -> Result<Self> {
        use crate::DType::*;
        match descr {
            Value::String(string) => Ok(Plain {
                ty: string,
                shape: vec![],
            }),
            Value::List(ref list) => Ok(Record(convert_list_to_record_fields(list)?)),
            _ => invalid_data("must be string or list"),
        }
    }
}

fn convert_list_to_record_fields(values: &[Value]) -> Result<Vec<Field>> {
    first_error(values.iter().map(|value| match *value {
        Value::List(ref tuple) => convert_tuple_to_record_field(tuple),
        _ => invalid_data("list must contain list or tuple"),
    }))
}

fn convert_tuple_to_record_field(tuple: &[Value]) -> Result<Field> {
    use self::Value::{List, String};

    match tuple.len() {
        2 | 3 => match (&tuple[0], &tuple[1], tuple.get(2)) {
            (String(name), String(dtype), ref shape) => Ok(Field {
                name: name.clone(),
                dtype: DType::Plain {
                    ty: dtype.clone(),
                    shape: if let &Some(s) = shape {
                        convert_value_to_shape(s)?
                    } else {
                        vec![]
                    },
                },
            }),
            (String(name), List(list), None) => Ok(Field {
                name: name.clone(),
                dtype: DType::Record(convert_list_to_record_fields(list)?),
            }),
            (&String(_), &List(_), Some(_)) => {
                invalid_data("nested arrays of Record types are not supported.")
            }
            _ => invalid_data("list entry must contain a string for id and a valid dtype"),
        },
        _ => invalid_data("list entry must contain 2 or 3 items"),
    }
}

fn convert_value_to_shape(field: &Value) -> Result<Vec<u64>> {
    if let Value::List(ref lengths) = *field {
        first_error(lengths.iter().map(convert_value_to_positive_integer))
    } else {
        invalid_data("shape must be list or tuple")
    }
}

fn convert_value_to_positive_integer(number: &Value) -> Result<u64> {
    if let Value::Integer(number) = *number {
        if number > 0 {
            Ok(number as u64)
        } else {
            invalid_data("number must be positive")
        }
    } else {
        invalid_data("must be a number")
    }
}

fn first_error<I, T>(results: I) -> Result<Vec<T>>
where
    I: IntoIterator<Item = Result<T>>,
{
    let mut vector = vec![];
    for result in results {
        vector.push(result?);
    }
    Ok(vector)
}

fn invalid_data<T>(message: &str) -> Result<T> {
    use std::io::{Error, ErrorKind};
    Err(Error::new(ErrorKind::InvalidData, message.to_string()))
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Value {
    String(String),
    Integer(i64),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
}

pub fn parse_header(bs: &[u8]) -> IResult<&[u8], Value> {
    parser::header(bs)
}

mod parser {
    use super::Value;
    use nom::{
        branch::alt,
        bytes::complete::{is_not, tag},
        character::complete::{char, digit1, space0},
        combinator::{map, map_res, opt},
        multi::{length_value, separated_list0},
        number::complete::le_u16,
        sequence::{delimited, separated_pair, terminated},
        IResult,
    };
    use std::str::{self, FromStr};

    use nom::Parser;
    type Res<'a, O> = IResult<&'a [u8], O>;

    fn ws<'a, F, O>(mut inner: F) -> impl FnMut(&'a [u8]) -> Res<'a, O>
    where
        F: 'a + Parser<&'a [u8], Output = O, Error = nom::error::Error<&'a [u8]>>,
    {
        move |input| delimited(space0, |i| inner.parse(i), space0).parse(input)
    }

    pub fn header(input: &[u8]) -> IResult<&[u8], Value> {
        let (input, _) = tag(&[0x93][..])(input)?;
        let (input, _) = tag(&b"NUMPY"[..])(input)?;
        let (input, version) = alt((tag(&[0x01, 0x00][..]), tag(&[0x02, 0x00][..]))).parse(input)?;
        let (input, val) = if version == &[0x01, 0x00] {
            length_value(le_u16, item).parse(input)?
        } else {
            use nom::number::complete::le_u32;
            length_value(le_u32, item).parse(input)?
        };
        Ok((input, val))
    }

    pub fn integer(input: &[u8]) -> IResult<&[u8], Value> {
        map_res(map_res(ws(digit1), str::from_utf8), |s| {
            i64::from_str(s).map(Value::Integer)
        })
        .parse(input)
    }

    pub fn boolean(input: &[u8]) -> IResult<&[u8], Value> {
        ws(alt((
            map(tag("True"), |_| Value::Bool(true)),
            map(tag("False"), |_| Value::Bool(false)),
        )))
        .parse(input)
    }

    pub fn string(input: &[u8]) -> IResult<&[u8], Value> {
        let quoted_string = alt((
            delimited(char('\"'), is_not("\""), char('\"')),
            delimited(char('\''), is_not("'"), char('\'')),
        ));

        map_res(ws(quoted_string), |s: &[u8]| {
            str::from_utf8(s).map(|s| Value::String(s.to_string()))
        })
        .parse(input)
    }

    fn list_inner(input: &'_ [u8]) -> Res<'_, Vec<Value>> {
        terminated(
            separated_list0(ws(tag(&b","[..])), item),
            opt(tag(&b","[..])),
        )
        .parse(input)
    }

    pub fn list(input: &[u8]) -> IResult<&[u8], Value> {
        let array = alt((
            delimited(ws(tag("[")), list_inner, ws(tag("]"))),
            delimited(ws(tag("(")), list_inner, ws(tag(")"))),
        ));

        map(array, Value::List).parse(input)
    }

    pub fn map_parser(input: &[u8]) -> IResult<&[u8], Value> {
        let comma = ws(tag(","));
        let colon = ws(tag(":"));

        let kv = separated_pair(
            map_res(string, |v| match v {
                Value::String(s) => Ok(s),
                _ => Err(()),
            }),
            colon,
            item,
        );

        let entries = separated_list0(comma, kv);

        let map_inner = terminated(entries, opt(tag(",")));

        map(
            delimited(ws(tag("{")), map_inner, ws(tag("}"))),
            |v: Vec<(String, Value)>| Value::Map(v.into_iter().collect()),
        )
        .parse(input)
    }

    pub fn item(input: &[u8]) -> IResult<&[u8], Value> {
        alt((integer, boolean, string, list, map_parser)).parse(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn description_of_record_array_as_python_list_of_tuples() {
        let dtype = DType::Record(vec![
            Field {
                name: "float".to_string(),
                dtype: DType::Plain {
                    ty: ">f4".to_string(),
                    shape: vec![],
                },
            },
            Field {
                name: "byte".to_string(),
                dtype: DType::Plain {
                    ty: "<u1".to_string(),
                    shape: vec![],
                },
            },
        ]);
        let expected = "[('float', '>f4'), ('byte', '<u1'), ]";
        assert_eq!(dtype.descr(), expected);
    }

    #[test]
    fn description_of_unstructured_primitive_array() {
        let dtype = DType::Plain {
            ty: ">f8".to_string(),
            shape: vec![],
        };
        assert_eq!(dtype.descr(), "'>f8'");
    }

    #[test]
    fn description_of_nested_record_dtype() {
        let dtype = DType::Record(vec![Field {
            name: "parent".to_string(),
            dtype: DType::Record(vec![Field {
                name: "child".to_string(),
                dtype: DType::Plain {
                    ty: "<i4".to_string(),
                    shape: vec![],
                },
            }]),
        }]);
        assert_eq!(dtype.descr(), "[('parent', [('child', '<i4'), ]), ]");
    }

    #[test]
    fn converts_simple_description_to_record_dtype() {
        let dtype = ">f8".to_string();
        assert_eq!(
            DType::from_descr(Value::String(dtype.clone())).unwrap(),
            DType::Plain {
                ty: dtype,
                shape: vec![]
            }
        );
    }

    #[test]
    fn converts_record_description_to_record_dtype() {
        let descr = parse("[('a', '<u2'), ('b', '<f4')]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "a".to_string(),
                dtype: DType::Plain {
                    ty: "<u2".to_string(),
                    shape: vec![],
                },
            },
            Field {
                name: "b".to_string(),
                dtype: DType::Plain {
                    ty: "<f4".to_string(),
                    shape: vec![],
                },
            },
        ]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }

    #[test]
    fn record_description_with_onedimensional_field_shape_declaration() {
        let descr = parse("[('a', '>f8', (1,))]");
        let expected_dtype = DType::Record(vec![Field {
            name: "a".to_string(),
            dtype: DType::Plain {
                ty: ">f8".to_string(),
                shape: vec![1],
            },
        }]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }

    #[test]
    fn record_description_with_nested_record_field() {
        let descr = parse("[('parent', [('child', '<i4')])]");
        let expected_dtype = DType::Record(vec![Field {
            name: "parent".to_string(),
            dtype: DType::Record(vec![Field {
                name: "child".to_string(),
                dtype: DType::Plain {
                    ty: "<i4".to_string(),
                    shape: vec![],
                },
            }]),
        }]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
    }

    #[test]
    fn errors_on_nested_record_field_array() {
        let descr = parse("[('parent', [('child', '<i4')], (2,))]");
        assert!(DType::from_descr(descr).is_err());
    }

    #[test]
    fn errors_on_value_variants_that_cannot_be_converted() {
        let no_dtype = Value::Bool(false);
        assert!(DType::from_descr(no_dtype).is_err());
    }

    #[test]
    fn errors_when_record_list_does_not_contain_lists() {
        let faulty_list = parse("['a', 123]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_few_items() {
        let faulty_list = parse("[('a')]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_many_items() {
        let faulty_list = parse("[('a', 1, 2, 3)]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_non_strings_for_id_or_dtype() {
        let faulty_list = parse("[(1, 2)]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_shape_is_not_a_list() {
        let no_shape = parse("1");
        assert!(convert_value_to_shape(&no_shape).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_not_a_number() {
        let no_number = parse("[]");
        assert!(convert_value_to_positive_integer(&no_number).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_not_positive() {
        assert!(convert_value_to_positive_integer(&parse("0")).is_err());
    }

    fn parse(source: &str) -> Value {
        parser::item(source.as_bytes())
            .expect("could not parse Python expression")
            .1
    }
}




