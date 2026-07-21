use crate::error::PassError;
use crate::parser::ast::{AstFunction, AstGenericParam, AstModule, AstStructDef, AstType, Variance};

pub struct VarianceChecker;

impl VarianceChecker {
    pub fn new() -> Self {
        Self
    }

    pub fn run(&self, module: &AstModule) -> Result<(), PassError> {
        for func in &module.functions {
            self.check_function_variance(func)?;
        }
        for s in &module.structs {
            self.check_struct_variance(s)?;
        }
        for impl_def in &module.impls {
            for method in &impl_def.methods {
                self.check_function_variance(method)?;
            }
        }
        Ok(())
    }

    fn check_function_variance(&self, func: &AstFunction) -> Result<(), PassError> {
        for param in &func.type_params {
            if let AstGenericParam::Type(name, _, variance) | AstGenericParam::Hkt(name, _, _, variance) = param {
                let name = name.as_str();
                match variance {
                    Variance::Covariant => {
                        let in_ret = self.type_uses_param(&func.return_ty, name);
                        let in_params = func.params.iter().any(|p| self.type_uses_param(&p.ty, name));
                        if in_params && !in_ret {
                            return Err(PassError::TypeError {
                                func: func.name.name.clone(),
                                detail: format!(
                                    "covariant type parameter `{name}` appears only in parameter position — consider using invariant or contravariant"
                                ),
                            });
                        }
                    }
                    Variance::Contravariant => {
                        let in_ret = self.type_uses_param(&func.return_ty, name);
                        let in_params = func.params.iter().any(|p| self.type_uses_param(&p.ty, name));
                        if in_ret && !in_params {
                            return Err(PassError::TypeError {
                                func: func.name.name.clone(),
                                detail: format!(
                                    "contravariant type parameter `{name}` appears only in return position — consider using invariant or covariant"
                                ),
                            });
                        }
                    }
                    Variance::Invariant => {}
                }
            }
        }
        Ok(())
    }

    fn check_struct_variance(&self, s: &AstStructDef) -> Result<(), PassError> {
        for param in &s.type_params {
            if let AstGenericParam::Type(name, _, variance) | AstGenericParam::Hkt(name, _, _, variance) = param {
                let name = name.as_str();
                match variance {
                    Variance::Covariant => {
                        for field in &s.fields {
                            if let AstType::Fn { params, ret, .. } = &field.ty {
                                if self.type_uses_param(&ret, name) {
                                    // In return type of a fn-typed field — positive position, ok
                                }
                                if params.iter().any(|p| self.type_uses_param(p, name)) {
                                    return Err(PassError::TypeError {
                                        func: s.name.name.clone(),
                                        detail: format!(
                                            "covariant type parameter `{name}` appears in contravariant position (parameter of fn-typed field `{}`)",
                                            field.name.name
                                        ),
                                    });
                                }
                            }
                        }
                    }
                    Variance::Contravariant => {
                        for field in &s.fields {
                            if let AstType::Fn { params: _, ret, .. } = &field.ty {
                                if self.type_uses_param(&ret, name) {
                                    return Err(PassError::TypeError {
                                        func: s.name.name.clone(),
                                        detail: format!(
                                            "contravariant type parameter `{name}` appears in covariant position (return of fn-typed field `{}`)",
                                            field.name.name
                                        ),
                                    });
                                }
                            } else if self.type_uses_param(&field.ty, name) {
                                return Err(PassError::TypeError {
                                    func: s.name.name.clone(),
                                    detail: format!(
                                        "contravariant type parameter `{name}` in field `{}` — contravariant params should only appear in fn parameter position",
                                        field.name.name
                                    ),
                                });
                            }
                        }
                    }
                    Variance::Invariant => {}
                }
            }
        }
        Ok(())
    }

    fn type_uses_param(&self, ty: &AstType, param_name: &str) -> bool {
        match ty {
            AstType::Named(n, _) => n == param_name,
            AstType::AssocType { base, .. } => base == param_name,
            AstType::Generic { name, args, .. } => {
                name == param_name || args.iter().any(|a| self.type_uses_param(a, param_name))
            }
            AstType::Option(inner, _) => self.type_uses_param(inner, param_name),
            AstType::Result(ok, err, _) => {
                self.type_uses_param(ok, param_name) || self.type_uses_param(err, param_name)
            }
            AstType::List(inner, _) => self.type_uses_param(inner, param_name),
            AstType::Map(k, v, _) => {
                self.type_uses_param(k, param_name) || self.type_uses_param(v, param_name)
            }
            AstType::Chan(inner, _) => self.type_uses_param(inner, param_name),
            AstType::Atomic(inner, _) => self.type_uses_param(inner, param_name),
            AstType::Mutex(inner, _) => self.type_uses_param(inner, param_name),
            AstType::Grad(inner, _) => self.type_uses_param(inner, param_name),
            AstType::Sparse(inner, _) => self.type_uses_param(inner, param_name),
            AstType::Fn { params, ret, .. } => {
                params.iter().any(|p| self.type_uses_param(p, param_name))
                    || self.type_uses_param(ret, param_name)
            }
            AstType::Tuple(elems, _) => elems.iter().any(|e| self.type_uses_param(e, param_name)),
            AstType::Array { elem, .. } => self.type_uses_param(elem, param_name),
            _ => false,
        }
    }
}

impl Default for VarianceChecker {
    fn default() -> Self {
        Self::new()
    }
}
